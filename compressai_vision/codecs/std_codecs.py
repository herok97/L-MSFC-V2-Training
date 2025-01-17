# Copyright (c) 2022-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import configparser
import errno
import json
import logging
import math
import os
import sys
import time
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn

from compressai_vision.model_wrappers import BaseWrapper
from compressai_vision.registry import register_codec
from compressai_vision.utils import time_measure
from compressai_vision.utils.dataio import (
    PixelFormat,
    read_image_to_rgb_tensor,
    readwriteYUV,
)
from compressai_vision.utils.external_exec import run_cmdline, run_cmdlines_parallel

from .encdec_utils import *
from .utils import MIN_MAX_DATASET, min_max_inv_normalization, min_max_normalization


def get_filesize(filepath: Union[Path, str]) -> int:
    return Path(filepath).stat().st_size


# TODO (fracape) belongs to somewhere else?
def load_bitstream(path):
    with open(path, "rb") as fd:
        buf = BytesIO(fd.read())

    return buf.getvalue()


@register_codec("vtm")
class VTM(nn.Module):
    """Encoder/Decoder class for VVC - VTM reference software"""

    def __init__(
        self,
        vision_model: BaseWrapper,
        dataset: Dict,
        **kwargs,
    ):
        super().__init__()

        self.enc_cfgs = kwargs["encoder_config"]
        codec_paths = kwargs["codec_paths"]

        self.encoder_path = Path(codec_paths["encoder_exe"])
        self.decoder_path = Path(codec_paths["decoder_exe"])
        self.cfg_file = Path(codec_paths["cfg_file"])

        self.parcat_path = Path(codec_paths["parcat_exe"])  # optional
        self.parallel_encoding = self.enc_cfgs["parallel_encoding"]  # parallel option
        self.hash_check = self.enc_cfgs["hash_check"]  # md5 hash check
        self.stash_outputs = self.enc_cfgs["stash_outputs"]

        check_list_of_paths = [self.encoder_path, self.decoder_path, self.cfg_file]
        if self.parallel_encoding:  # miminum
            check_list_of_paths.append(self.parcat_path)

        for file_path in check_list_of_paths:
            if not file_path.is_file():
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), file_path
                )

        self.qp = self.enc_cfgs["qp"]
        self.eval_encode = kwargs["eval_encode"]

        self.dump = kwargs["dump"]
        self.fpn_sizes_json_dump = self.dump["fpn_sizes_json_dump"]
        self.vision_model = vision_model

        self.datacatalog = dataset.datacatalog
        self.dataset_name = dataset.config["dataset_name"]

        if self.datacatalog in MIN_MAX_DATASET:
            self.min_max_dataset = MIN_MAX_DATASET[self.datacatalog]
        elif self.dataset_name in MIN_MAX_DATASET:
            self.min_max_dataset = MIN_MAX_DATASET[self.dataset_name]
        else:
            raise ValueError("dataset not recognized for normalization")

        self.yuvio = readwriteYUV(device="cpu", format=PixelFormat.YUV400_10le)

        self.intra_period = self.enc_cfgs["intra_period"]
        self.frame_rate = 1
        if not self.datacatalog == "MPEGOIV6":
            config = configparser.ConfigParser()
            config.read(f"{dataset['config']['root']}/{dataset['config']['seqinfo']}")
            self.frame_rate = config["Sequence"]["frameRate"]

        self.logger = logging.getLogger(self.__class__.__name__)
        self.verbosity = kwargs["verbosity"]
        self.ffmpeg_loglevel = "error"
        logging_level = logging.WARN
        if self.verbosity == 1:
            logging_level = logging.INFO
        if self.verbosity >= 2:
            logging_level = logging.DEBUG
            self.ffmpeg_loglevel = "debug"

        self.logger.setLevel(logging_level)

    # can be added to base class (if inherited) | Should we inherit from the base codec?
    @property
    def qp_value(self):
        return self.qp

    # can be added to base class (if inherited) | Should we inherit from the base codec?
    @property
    def eval_encode_type(self):
        return self.eval_encode

    def get_encode_cmd(
        self,
        inp_yuv_path: Path,
        qp: int,
        bitstream_path: Path,
        width: int,
        height: int,
        nb_frames: int = 1,
        parallel_encoding: bool = False,
        hash_check: int = 0,
        chroma_format: str = "400",
        input_bitdepth: int = 10,
        output_bitdepth: int = 0,
    ) -> List[Any]:
        level = 5.1 if nb_frames > 1 else 6.2  # according to MPEG's anchor
        if output_bitdepth == 0:
            output_bitdepth = input_bitdepth

        decodingRefreshType = 1 if self.intra_period >= 1 else 0
        base_cmd = [
            self.encoder_path,
            "-i",
            inp_yuv_path,
            "-c",
            self.cfg_file,
            "-q",
            qp,
            "-o",
            "/dev/null",
            "-wdt",
            width,
            "-hgt",
            height,
            "-fr",
            self.frame_rate,
            "-ts",  # temporal subsampling to prevent default period of 8 in all intra
            "1",
            "-v",
            "6",
            f"--Level={level}",
            f"--IntraPeriod={self.intra_period}",
            f"--InputChromaFormat={chroma_format}",
            f"--InputBitDepth={input_bitdepth}",
            f"--InternalBitDepth={output_bitdepth}",
            "--ConformanceWindowMode=1",  # needed?
            "-dph",  # md5 has,
            hash_check,
            f"--DecodingRefreshType={decodingRefreshType}",
        ]

        if parallel_encoding is False or nb_frames <= (self.intra_period + 1):
            base_cmd.append(f"--BitstreamFile={bitstream_path}")
            base_cmd.append(f"--FramesToBeEncoded={nb_frames}")
            cmd = list(map(str, base_cmd))
            self.logger.debug(cmd)
            return [cmd]

        num_parallels = round((nb_frames / self.intra_period) + 0.5)

        list_of_num_of_frameSkip = []
        list_of_num_of_framesToBeEncoded = []
        total_num_frames_to_code = nb_frames

        frameSkip = 0
        nb_framesToBeEncoded = self.intra_period + 1
        for _ in range(num_parallels):
            list_of_num_of_frameSkip.append(frameSkip)

            nb_framesToBeEncoded = min(total_num_frames_to_code, nb_framesToBeEncoded)
            list_of_num_of_framesToBeEncoded.append(nb_framesToBeEncoded)

            frameSkip += self.intra_period
            total_num_frames_to_code -= self.intra_period

        bitstream_path_p = Path(bitstream_path).parent
        file_stem = Path(bitstream_path).stem
        ext = Path(bitstream_path).suffix

        parallel_cmds = []
        for e, items in enumerate(
            zip(list_of_num_of_frameSkip, list_of_num_of_framesToBeEncoded)
        ):
            frameSkip, framesToBeEncoded = items
            sbitstream_path = (
                str(bitstream_path_p)
                + "/"
                + str(file_stem)
                + f"-part-{e:03d}"
                + str(ext)
            )

            pcmd = deepcopy(base_cmd)
            pcmd.append(f"--BitstreamFile={sbitstream_path}")
            pcmd.append(f"--FrameSkip={frameSkip}")
            pcmd.append(f"--FramesToBeEncoded={framesToBeEncoded}")

            cmd = list(map(str, pcmd))
            self.logger.debug(cmd)

            parallel_cmds.append(cmd)

        return parallel_cmds

    def get_parcat_cmd(
        self,
        bitstream_path: str,
    ) -> List[Any]:
        pdir = Path(bitstream_path).parent
        fstem = Path(bitstream_path).stem
        ext = str(Path(bitstream_path).suffix)

        bitstream_lists = sorted(Path(pdir).glob(f"{fstem}-part-*{ext}"))

        cmd = [self.parcat_path]
        for bpath in bitstream_lists:
            cmd.append(str(bpath))
        cmd.append(bitstream_path)

        cmd = list(map(str, cmd))
        self.logger.debug(cmd)
        return cmd, bitstream_lists

    def get_decode_cmd(
        self, yuv_dec_path: Path, bitstream_path: Path, output_bitdepth: int = 10
    ) -> List[Any]:
        cmd = [
            self.decoder_path,
            "-b",
            bitstream_path,
            "-o",
            yuv_dec_path,
            "-d",
            output_bitdepth,
        ]
        cmd = list(map(str, cmd))
        self.logger.debug(cmd)
        return cmd

    def convert_input_to_yuv(self, input: Dict, file_prefix: str):
        nb_frames = 1
        file_names = input["file_names"]
        if len(file_names) > 1:  # video
            # NOTE: using glob for now, should be more robust and look at skipped
            # NOTE: somewhat rigid pattern (lowercase png)
            filename_pattern = f"{str(Path(file_names[0]).parent)}/*.png"
            nb_frames = input["last_frame"] - input["frame_skip"]
            images_in_folder = len(
                [file for file in Path(file_names[0]).parent.glob("*.png")]
            )
            assert (
                images_in_folder == nb_frames
            ), f"input folder contains {images_in_folder} images, {nb_frames} were expected"

            input_info = [
                "-pattern_type",
                "glob",
                "-i",
                filename_pattern,
                # "-start_number",
                # "0",  # warning, start frame is 0 for now
                # "-vframes",
                # f"{nb_frames}",
            ]
        else:
            input_info = ["-i", file_names[0]]

        chroma_format = self.enc_cfgs["chroma_format"]
        input_bitdepth = self.enc_cfgs["input_bitdepth"]

        frame_width = math.ceil(input["org_input_size"]["width"] / 2) * 2
        frame_height = math.ceil(input["org_input_size"]["height"] / 2) * 2
        file_prefix = f"{file_prefix}_{frame_width}x{frame_height}_{self.frame_rate}fps_{input_bitdepth}bit_p{chroma_format}"
        yuv_in_path = f"{file_prefix}_input.yuv"

        pix_fmt_suffix = "10le" if input_bitdepth == 10 else ""
        chroma_format = "gray" if chroma_format == "400" else f"yuv{chroma_format}p"

        # TODO (fracape)
        # we don't enable skipping frames (codec.skip_n_frames) nor use n_frames_to_be_encoded in video mode

        convert_cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            f"{self.ffmpeg_loglevel}",
        ]
        convert_cmd += input_info
        convert_cmd += [
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-f",
            "rawvideo",
            "-pix_fmt",
            f"{chroma_format}{pix_fmt_suffix}",
        ]

        convert_cmd.append(yuv_in_path)

        run_cmdline(convert_cmd)

        return (yuv_in_path, nb_frames, frame_width, frame_height, file_prefix)

    def encode(
        self,
        x: Dict,
        codec_output_dir,
        bitstream_name,
        file_prefix: str = "",
        img_input=False,
    ) -> bool:

        bitdepth = 10  # TODO (fracape) (add this as config)

        if file_prefix == "":
            file_prefix = f"{codec_output_dir}/{bitstream_name}"
        else:
            file_prefix = f"{codec_output_dir}/{bitstream_name}-{file_prefix}"

        print(f"\n-- encoding ${file_prefix}", file=sys.stdout)

        if img_input:
            (yuv_in_path, nb_frames, frame_width, frame_height, file_prefix) = (
                self.convert_input_to_yuv(input=x, file_prefix=file_prefix)
            )

        else:
            (
                frames,
                self.feature_size,
                self.subframe_heights,
            ) = self.vision_model.reshape_feature_pyramid_to_frame(
                x["data"], packing_all_in_one=True
            )

            # Generate json files with fpn sizes for the decoder
            # manually activate the following and run in encode_only mode
            if self.fpn_sizes_json_dump:
                self.dump_fpn_sizes_json(file_prefix, bitstream_name, codec_output_dir)

            minv, maxv = self.min_max_dataset
            frames, mid_level = min_max_normalization(
                frames, minv, maxv, bitdepth=bitdepth
            )

            nb_frames, frame_height, frame_width = frames.size()
            input_bitdepth = self.enc_cfgs["input_bitdepth"]
            chroma_format = self.enc_cfgs["chroma_format"]
            file_prefix = f"{file_prefix}_{frame_width}x{frame_height}_{self.frame_rate }fps_{input_bitdepth}bit_p{chroma_format}"

            yuv_in_path = f"{file_prefix}_input.yuv"

            self.yuvio.setWriter(
                write_path=yuv_in_path,
                frmWidth=frame_width,
                frmHeight=frame_height,
            )

            for frame in frames:
                self.yuvio.write_one_frame(frame, mid_level=mid_level)

        bitstream_path = f"{file_prefix}.bin"
        logpath = Path(f"{file_prefix}_enc.log")
        cmds = self.get_encode_cmd(
            yuv_in_path,
            width=frame_width,
            height=frame_height,
            qp=self.qp,
            bitstream_path=bitstream_path,
            nb_frames=nb_frames,
            chroma_format=self.enc_cfgs["chroma_format"],
            input_bitdepth=self.enc_cfgs["input_bitdepth"],
            output_bitdepth=self.enc_cfgs["output_bitdepth"],
            parallel_encoding=self.parallel_encoding,
            hash_check=self.hash_check,
        )
        # self.logger.debug(cmd)

        start = time.time()
        if len(cmds) > 1:  # post parallel encoding
            run_cmdlines_parallel(cmds, logpath=logpath)
        else:
            run_cmdline(cmds[0], logpath=logpath)
        enc_time = time.time() - start
        self.logger.debug(f"enc_time:{enc_time}")

        if len(cmds) > 1:  # post parallel encoding
            cmd, list_of_bitstreams = self.get_parcat_cmd(bitstream_path)
            run_cmdline(cmd)

            if self.stash_outputs:
                for partial in list_of_bitstreams:
                    Path(partial).unlink()

        assert Path(
            bitstream_path
        ).is_file(), f"bitstream {bitstream_path} was not created"

        if not img_input:
            inner_codec_bitstream = load_bitstream(bitstream_path)

            # Bistream header to make bitstream self-decodable
            _ = self.write_n_bit(self.bitdepth)
            _ = self.write_rft_chSize(x["chSize"])
            _ = self.write_packed_frame_size((frame_height, frame_width))
            _ = self.write_min_max_values()

            pre_info_bitstream = self.get_io_buffer_contents()
            bitstream = pre_info_bitstream + inner_codec_bitstream

            with open(bitstream_path, "wb") as fw:
                fw.write(bitstream)

        if not self.dump["dump_yuv_input"]:
            Path(yuv_in_path).unlink()

        # to be compatible with the pipelines
        # per frame bits can be collected by parsing enc log to be more accurate
        avg_bytes_per_frame = get_filesize(bitstream_path) / nb_frames
        all_bytes_per_frame = [avg_bytes_per_frame] * nb_frames

        return {
            "bytes": all_bytes_per_frame,
            "bitstream": bitstream_path,
        }

    def decode(
        self,
        bitstream_path: Path = None,
        codec_output_dir: str = "",
        file_prefix: str = "",
        org_img_size: Dict = None,
        img_input=False,
    ) -> bool:
        del org_img_size

        bitstream_path = Path(bitstream_path)
        assert bitstream_path.is_file()

        output_file_prefix = bitstream_path.stem

        dec_path = codec_output_dir / "dec"
        dec_path.mkdir(parents=True, exist_ok=True)
        logpath = Path(f"{dec_path}/{output_file_prefix}_dec.log")

        if img_input:
            video_info = get_raw_video_file_info(output_file_prefix.split("qp")[-1])
            frame_width = video_info["width"]
            frame_height = video_info["height"]
            yuv_dec_path = f"{dec_path}/{output_file_prefix}_dec.yuv"
            cmd = self.get_decode_cmd(
                bitstream_path=bitstream_path,
                yuv_dec_path=yuv_dec_path,
                output_bitdepth=video_info["bitdepth"],
            )
            # self.logger.debug(cmd)
        else:
            bitstream = load_bitstream(bitstream_path)
            n_bit = self.read_n_bit(bitstream)
            chH, chW = self.read_rft_chSize(bitstream)
            frmH, frmW = self.read_packed_frame_size(bitstream)
            _min_max_buffer = self.read_min_max_values(bitstream)
            with open(bitstream_path, "wb") as fw:
                fw.write(bitstream.read())
            bitstream_path_tm = f"{file_prefix}_tmp.bin"
            cmd = self.get_decode_cmd(
                bitstream_path=bitstream_path,
                yuv_dec_path=yuv_dec_path,
                output_bitdepth=video_info["bitdepth"],
            )

        start = time_measure()
        run_cmdline(cmd, logpath=logpath)
        dec_time = time_measure() - start
        self.logger.debug(f"dec_time:{dec_time}")

        if img_input:

            # TODO assumes 8bit 420
            convert_cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-s",
                f"{frame_width}x{frame_height}",
                "-pix_fmt",
                "yuv420p",
                "-i",
                yuv_dec_path,
            ]

            # not cropping for now
            # crop_cmd = ["-vf", f"crop={org_img_size['width']}:{org_img_size['height']}"]
            # convert_cmd += [crop_cmd]

            # TODO (fracape) hacky, clean this
            if self.datacatalog == "MPEGOIV6":
                output_png = f"{dec_path}/{output_file_prefix}.png"
            elif self.datacatalog == "SFUHW":
                prefix = output_file_prefix.split("qp")[0]
                output_png = f"{dec_path}/{prefix}%03d.png"
                convert_cmd += ["-start_number", "0"]
            elif self.datacatalog in ["MPEGHIEVE"]:
                convert_cmd += ["-start_number", "0"]
                output_png = f"{dec_path}/%06d.png"
            elif self.datacatalog in ["MPEGTVDTRACKING"]:
                convert_cmd += ["-start_number", "1"]
                output_png = f"{dec_path}/%06d.png"
            convert_cmd.append(output_png)

            run_cmdline(convert_cmd)

            rec_frames = []
            for file_path in sorted(dec_path.glob("*.png")):
                # rec_frames.append(read_image_to_rgb_tensor(file_path))
                rec_frames.append(str(file_path))

            # output the list of file paths for each frame
            # output = {"data": rec_frames}
            output = {"file_names": rec_frames}

        else:
            self.yuvio.setReader(
                read_path=yuv_dec_path,
                frmWidth=frame_width,
                frmHeight=frame_height,
            )

            nb_frames = get_filesize(yuv_dec_path) // (frame_width * frame_height * 2)

            rec_frames = []
            for i in range(nb_frames):
                rec_yuv = self.yuvio.read_one_frame(i)
                rec_frames.append(rec_yuv)

            rec_frames = torch.stack(rec_frames)

            minv, maxv = self.min_max_dataset
            rec_frames = min_max_inv_normalization(rec_frames, minv, maxv, bitdepth=10)

            # (fracape) should feature sizes be part of bitstream?
            thisdir = Path(__file__).parent
            if self.datacatalog == "MPEGOIV6":
                fpn_sizes = thisdir.joinpath(
                    f"../../data/mpeg-fcm/{self.datacatalog}/fpn-sizes/{self.dataset_name}/{file_prefix}.json"
                )
            else:
                fpn_sizes = thisdir.joinpath(
                    f"../../data/mpeg-fcm/{self.datacatalog}/fpn-sizes/{self.dataset_name}.json"
                )
            with fpn_sizes.open("r") as f:
                try:
                    json_dict = json.load(f)
                except json.decoder.JSONDecodeError as err:
                    print(f'Error reading file "{fpn_sizes}"')
                    raise err

            features = self.vision_model.reshape_frame_to_feature_pyramid(
                rec_frames,
                json_dict["fpn"],
                json_dict["subframe_heights"],
                packing_all_in_one=True,
            )
            if not self.dump["dump_yuv_packing_dec"]:
                Path(yuv_dec_path).unlink()

            output = {"data": features}

        return output

    # Functions required in the context of FCM to write a header that enables a self decodable bitstream
    def write_n_bit(self, n_bit):
        # adhoc method, warning redundant information
        return write_uchars(self._temp_io_buffer, (n_bit,))

    def write_rft_chSize(self, chSize):
        # adhoc method
        return write_uints(self._temp_io_buffer, chSize)

    def write_packed_frame_size(self, frmSize):
        # adhoc method, warning redundant information
        return write_uints(self._temp_io_buffer, frmSize)

    def write_min_max_values(self):
        # adhoc method to make bitstream self-decodable
        byte_cnts = write_uints(self._temp_io_buffer, (self.get_minmax_buffer_size(),))
        for min_max in self._min_max_buffer:
            byte_cnts += write_float32(self._temp_io_buffer, min_max)

        return byte_cnts

    def read_n_bit(self, fd):
        # adhoc method, warning redundant information
        return read_uchars(fd, 1)[0]

    def read_rft_chSize(self, fd):
        # adhoc method,
        return read_uints(fd, 2)

    def read_packed_frame_size(self, fd):
        # adhoc method, warning redundant information
        return read_uints(fd, 2)

    def read_min_max_values(self, fd):
        # adhoc method to make bitstream self-decodable
        num_minmax_pairs = read_uints(fd, 1)[0]

        min_max_buffer = []
        for _ in range(num_minmax_pairs):
            min_max = read_float32(fd, 2)
            min_max_buffer.append(min_max)

        return min_max_buffer

    def dump_fpn_sizes_json(self, file_prefix, bitstream_name, codec_output_dir):
        filename = file_prefix if file_prefix != "" else bitstream_name.split("_qp")[0]
        fpn_sizes_json = codec_output_dir / f"{filename}.json"
        with fpn_sizes_json.open("wb") as f:
            output = {
                "fpn": self.feature_size,
                "subframe_heights": self.subframe_heights,
            }
            f.write(json.dumps(output, indent=4).encode())
        print(f"fpn sizes json dump generated, exiting")
        raise SystemExit(0)


@register_codec("hm")
class HM(VTM):
    """Encoder / Decoder class for HEVC - HM reference software"""

    def __init__(
        self,
        vision_model: BaseWrapper,
        dataset: Dict,
        **kwargs,
    ):
        super().__init__(vision_model, dataset, **kwargs)

    def get_encode_cmd(
        self,
        inp_yuv_path: Path,
        qp: int,
        bitstream_path: Path,
        width: int,
        height: int,
        nb_frames: int = 1,
        parallel_encoding: bool = False,
        hash_check: int = 0,
        chroma_format: str = "400",
        input_bitdepth: int = 10,
        output_bitdepth: int = 0,
    ) -> List[Any]:
        level = 5.1 if nb_frames > 1 else 6.2  # according to MPEG's anchor
        if output_bitdepth == 0:
            output_bitdepth = input_bitdepth

        decodingRefreshType = 1 if self.intra_period >= 1 else 0
        base_cmd = [
            self.encoder_path,
            "-i",
            inp_yuv_path,
            "-c",
            self.cfg_file,
            "-q",
            qp,
            "-o",
            "/dev/null",
            "-wdt",
            width,
            "-hgt",
            height,
            "-fr",
            self.frame_rate,
            "-ts",  # temporal subsampling to prevent default period of 8 in all intra
            "1",
            f"--Level={level}",
            f"--IntraPeriod={self.intra_period}",
            f"--InputChromaFormat={chroma_format}",
            f"--InputBitDepth={input_bitdepth}",
            f"--InternalBitDepth={output_bitdepth}",
            "--ConformanceWindowMode=1",  # needed?
            f"--DecodingRefreshType={decodingRefreshType}",
        ]

        if parallel_encoding is False or nb_frames <= (self.intra_period + 1):
            base_cmd.append(f"--BitstreamFile={bitstream_path}")
            base_cmd.append(f"--FramesToBeEncoded={nb_frames}")
            cmd = list(map(str, base_cmd))
            self.logger.debug(cmd)
            return [cmd]

        num_parallels = round((nb_frames / self.intra_period) + 0.5)

        list_of_num_of_frameSkip = []
        list_of_num_of_framesToBeEncoded = []
        total_num_frames_to_code = nb_frames

        frameSkip = 0
        nb_framesToBeEncoded = self.intra_period + 1
        for _ in range(num_parallels):
            list_of_num_of_frameSkip.append(frameSkip)

            nb_framesToBeEncoded = min(total_num_frames_to_code, nb_framesToBeEncoded)
            list_of_num_of_framesToBeEncoded.append(nb_framesToBeEncoded)

            frameSkip += self.intra_period
            total_num_frames_to_code -= self.intra_period

        bitstream_path_p = Path(bitstream_path).parent
        file_stem = Path(bitstream_path).stem
        ext = Path(bitstream_path).suffix

        parallel_cmds = []
        for e, items in enumerate(
            zip(list_of_num_of_frameSkip, list_of_num_of_framesToBeEncoded)
        ):
            frameSkip, framesToBeEncoded = items
            sbitstream_path = (
                str(bitstream_path_p)
                + "/"
                + str(file_stem)
                + f"-part-{e:03d}"
                + str(ext)
            )

            pcmd = deepcopy(base_cmd)
            pcmd.append(f"--BitstreamFile={sbitstream_path}")
            pcmd.append(f"--FrameSkip={frameSkip}")
            pcmd.append(f"--FramesToBeEncoded={framesToBeEncoded}")

            cmd = list(map(str, pcmd))
            self.logger.debug(cmd)

            parallel_cmds.append(cmd)

        return parallel_cmds


@register_codec("vvenc")
class VVENC(VTM):
    """Encoder / Decoder class for VVC - vvenc/vvdec  software"""

    def __init__(
        self,
        vision_model: BaseWrapper,
        dataset_name: "str" = "",
        **kwargs,
    ):
        super().__init__(vision_model, dataset_name, **kwargs)

    def get_encode_cmd(
        self,
        inp_yuv_path: Path,
        qp: int,
        bitstream_path: Path,
        width: int,
        height: int,
        nb_frames: int = 1,
    ) -> List[Any]:
        cmd = [
            self.encoder_path,
            "-i",
            inp_yuv_path,
            "-q",
            qp,
            "--output",
            bitstream_path,
            "--size",
            f"{width}x{height}",
            "--framerate",
            self.frame_rate,
            "--frames",
            nb_frames,
            "--format",
            "yuv420_10",
            "--preset",
            "fast",
        ]
        return list(map(str, cmd))
