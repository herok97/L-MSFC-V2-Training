# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import struct
from pathlib import Path


def get_downsampled_shape(height, width, p):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    return int(new_h / p + 0.5), int(new_w / p + 0.5)


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def write_bit(fd, value):
    if value != 0 and value != 1:
        raise ValueError("Value must be 0 or 1")
    byte = value << 7
    fd.write(struct.pack("B", byte))


def read_bit(fd):
    byte = struct.unpack("B", fd.read(1))[0]
    return byte >> 7


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def encode_feature(nbframes, height, width, y_string, z_string, output, stream):
    y_string_length = len(y_string)
    z_string_length = len(z_string)

    if stream is None:  # with header (height, width)
        stream = Path(output).open("wb")
        write_uints(stream, (nbframes, height, width, y_string_length, z_string_length))
    else:
        write_uints(stream, (y_string_length, z_string_length))

    write_bytes(stream, y_string)
    write_bytes(stream, z_string)
    return stream


def decode_feature(inputpath, stream):
    if stream is None:
        stream = Path(inputpath).open("rb")
        header = read_uints(stream, 5)
        nbframes = header[0]
        height = header[1]
        width = header[2]
        y_string_length = header[3]
        z_string_length = header[4]
        y_string = [read_bytes(stream, y_string_length)]
        z_string = [read_bytes(stream, z_string_length)]
        return nbframes, height, width, y_string, z_string, stream
    else:
        header = read_uints(stream, 2)
        y_string_length = header[0]
        z_string_length = header[1]
        y_string = [read_bytes(stream, y_string_length)]
        z_string = [read_bytes(stream, z_string_length)]
        return None, None, None, y_string, z_string, stream


def encode_p_feature(
    height, width, y_string, z_string, mv_y_string, mv_z_string, output, stream
):
    mv_y_string_length = len(mv_y_string)
    mv_z_string_length = len(mv_z_string)
    y_string_length = len(y_string)
    z_string_length = len(z_string)

    if stream is None:  # with header (height, width)
        stream = Path(output).open("wb")
        write_uints(
            stream,
            (
                height,
                width,
                mv_y_string_length,
                mv_z_string_length,
                y_string_length,
                z_string_length,
            ),
        )
    else:
        write_uints(
            stream,
            (mv_y_string_length, mv_z_string_length, y_string_length, z_string_length),
        )

    write_bytes(stream, mv_y_string)
    write_bytes(stream, mv_z_string)
    write_bytes(stream, y_string)
    write_bytes(stream, z_string)
    return stream


def decode_p_feature(inputpath, stream):
    if stream is None:
        stream = Path(inputpath).open("rb")
        header = read_uints(stream, 6)
        height = header[0]
        width = header[1]
        mv_y_string_length = header[2]
        mv_z_string_length = header[3]
        y_string_length = header[4]
        z_string_length = header[5]

        mv_y_string = read_bytes(stream, mv_y_string_length)
        mv_z_string = read_bytes(stream, mv_z_string_length)
        y_string = [read_bytes(stream, y_string_length)]
        z_string = [read_bytes(stream, z_string_length)]
        return height, width, mv_y_string, mv_z_string, y_string, z_string, stream

    else:
        header = read_uints(stream, 4)
        mv_y_string_length = header[0]
        mv_z_string_length = header[1]
        y_string_length = header[2]
        z_string_length = header[3]
        mv_y_string = [read_bytes(stream, mv_y_string_length)]
        mv_z_string = [read_bytes(stream, mv_z_string_length)]

        y_string = [read_bytes(stream, y_string_length)]
        z_string = [read_bytes(stream, z_string_length)]
        return None, None, mv_y_string, mv_z_string, y_string, z_string, stream


# MODE DECISION
# def encode_feature(height, width, y_string, z_string, output, stream, mode):
#     y_string_length = len(y_string)
#     z_string_length = len(z_string)

#     if stream is None:  # with header (height, width)
#         stream = Path(output).open("wb")
#         write_bit(stream, 1 if mode == 'inter' else 0)
#         write_uints(stream, (height, width, y_string_length, z_string_length))
#     else:
#         write_bit(stream, 1 if mode == 'inter' else 0)
#         write_uints(stream, (y_string_length, z_string_length))

#     write_bytes(stream, y_string)
#     write_bytes(stream, z_string)
#     return stream

# def decode_feature(inputpath, stream):
#     if stream is None:
#         stream = Path(inputpath).open("rb")
#         mode = read_bit(stream)
#         if not mode:
#             header = read_uints(stream, 4)
#             height = header[0]
#             width = header[1]
#             y_string_length = header[2]
#             z_string_length = header[3]
#             y_string = [read_bytes(stream, y_string_length)]
#             z_string = [read_bytes(stream, z_string_length)]
#             return height, width, y_string, z_string, stream
#         else:
#             header = read_uints(stream, 6)
#             height = header[0]
#             width = header[1]
#             mv_y_string_length = header[2]
#             mv_z_string_length = header[3]
#             y_string_length = header[4]
#             z_string_length = header[5]

#             mv_y_string = read_bytes(stream, mv_y_string_length)
#             mv_z_string = read_bytes(stream, mv_z_string_length)
#             y_string = [read_bytes(stream, y_string_length)]
#             z_string = [read_bytes(stream, z_string_length)]
#             return height, width, mv_y_string, mv_z_string, y_string, z_string, stream
#     else:
#         mode = read_bit(stream)
#         if not mode:
#             header = read_uints(stream, 2)
#             y_string_length = header[0]
#             z_string_length = header[1]
#             y_string = [read_bytes(stream, y_string_length)]
#             z_string = [read_bytes(stream, z_string_length)]
#             return None, None, y_string, z_string, stream
#         else:
#             header = read_uints(stream, 4)
#             mv_y_string_length = header[0]
#             mv_z_string_length = header[1]
#             y_string_length = header[2]
#             z_string_length = header[3]
#             mv_y_string = [read_bytes(stream, mv_y_string_length)]
#             mv_z_string = [read_bytes(stream, mv_z_string_length)]

#             y_string = [read_bytes(stream, y_string_length)]
#             z_string = [read_bytes(stream, z_string_length)]
#             return None, None, mv_y_string, mv_z_string, y_string, z_string, stream

# def encode_p_feature(height, width, y_string, z_string, mv_y_string, mv_z_string, output, stream, mode):
#     mv_y_string_length = len(mv_y_string)
#     mv_z_string_length = len(mv_z_string)
#     y_string_length = len(y_string)
#     z_string_length = len(z_string)

#     if stream is None:  # with header (height, width)
#         stream = Path(output).open("wb")
#         write_bit(stream, 1 if mode == 'inter' else 0)
#         write_uints(stream, (height, width, mv_y_string_length, mv_z_string_length, y_string_length, z_string_length))
#     else:
#         write_bit(stream, 1 if mode == 'inter' else 0)
#         write_uints(stream, (mv_y_string_length, mv_z_string_length, y_string_length, z_string_length))

#     write_bytes(stream, mv_y_string)
#     write_bytes(stream, mv_z_string)
#     write_bytes(stream, y_string)
#     write_bytes(stream, z_string)
#     return stream

# def decode_p_feature(inputpath, stream, mode):
#     if stream is None:
#         stream = Path(inputpath).open("rb")
#         header = read_uints(stream, 6)
#         height = header[0]
#         width = header[1]
#         mv_y_string_length = header[2]
#         mv_z_string_length = header[3]
#         y_string_length = header[4]
#         z_string_length = header[5]

#         mv_y_string = read_bytes(stream, mv_y_string_length)
#         mv_z_string = read_bytes(stream, mv_z_string_length)
#         y_string = [read_bytes(stream, y_string_length)]
#         z_string = [read_bytes(stream, z_string_length)]
#         return height, width, mv_y_string, mv_z_string, y_string, z_string, stream

#     else:
#         header = read_uints(stream, 4)
#         mv_y_string_length = header[0]
#         mv_z_string_length = header[1]
#         y_string_length = header[2]
#         z_string_length = header[3]
#         mv_y_string = [read_bytes(stream, mv_y_string_length)]
#         mv_z_string = [read_bytes(stream, mv_z_string_length)]

#         y_string = [read_bytes(stream, y_string_length)]
#         z_string = [read_bytes(stream, z_string_length)]
#         return None, None, mv_y_string, mv_z_string, y_string, z_string, stream
