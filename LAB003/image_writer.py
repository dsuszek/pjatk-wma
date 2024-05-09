''' This script converts all .webp files in given directory into .png format and saves them
into other specified directory. Then it removes originals. 
This script was created in order to clean up rather messy downloads from Dalee image 
generator.
'''
from __future__ import annotations  # Type hints can now be used for methods returning their own class
import os  # Interactions with filesystem
import argparse  # Parsing terminal command parameters

import PIL
import pandas as pd
from PIL import Image  # Image processing
from collections.abc import Iterator  # Type hints


class ImageWriter:
    '''
    A class for writing images to a specified output path.

    Attributes:
        _IMAGE_FILENAME_FORMAT (str): The format string for generating image filenames.
        _output_path (str): The path where the images will be saved.
        _image_count (int): The current count of images written.
    '''

    _IMAGE_FILENAME_FORMAT: str = '{:03d}.'

    def __init__(self, output_path: str, grey_scale: str, output_format: str, output_width: int,
                 output_height: int) -> None:
        '''
        Initializes an ImageWriter instance.

        Args:
            output_path (str): The path where images will be saved.
            output_format (str): The format of output files.
        '''
        if not os.path.exists(output_path):
            print(f'Path {output_path} do not exists. Created path.')
            os.mkdir(output_path)

        self._output_path = output_path
        self._grey_scale = grey_scale
        self._output_format = output_format
        self._output_width = output_width
        self._output_height = output_height
        self._image_count = 1

        if self._output_format == 'jpg':
            self._output_format = 'jpeg'

        if output_format not in ('png', 'jpeg'):
            print('Invalid image file extension. Only .png and .jpeg extensions are allowed.')

        while os.path.exists(self._build_output_image_path()):
            self._image_count += 1
        print(f'Initial image count is {self._image_count - 1}.')

    def _build_output_image_path(self) -> str:
        '''
        Builds the full path for the next image to be written.

        Returns:
            str: The full path for the next image.
        '''
        return os.path.join(self._output_path,
                            ImageWriter._IMAGE_FILENAME_FORMAT.format(self._image_count) + self._output_format)

    def write_image(self, image: Image) -> None:
        '''
        Writes the provided image to the output path.

        Args:
            image (PIL.Image): The image to be written.
        '''
        try:
            image_path = self._build_output_image_path()
            image = image.resize((self._output_width, self._output_height))
            image = image.convert(self._grey_scale)
            if self._output_format not in ('png', 'jpeg'):
                return
            else:
                image.save(image_path, self._output_format)
            print(f'Saved image at {image_path}')
            self._image_count += 1
        except PIL.UnidentifiedImageError:
            print(f'Cannot identify image file at {image_path}')
        except AttributeError:
            print(f'Incorrect object type')


class ImageSource:
    '''
    A class for iterating over images in a specified source path.

    Attributes:
        _source_list (list): A list of paths to input images.
        _destructive (bool): A flag indicating whether images should be removed after iteration.
    '''

    def __init__(self, source_path: str) -> None:
        '''
        Initializes an ImageSource instance.

        Args:
            source_path (str): The path where input images are located.
            destructive (bool, optional): Flag indicating whether images should be removed after iteration. Defaults to False.
            recursive (bool, optional): Flag indicating whether to recursively search for images in subdirectories. Defaults to False.
        '''
        if not os.path.exists(source_path):
            raise OSError(f'Source path {source_path} do not exist.')
        self._source_list = self._generate_source_list(source_path)
        print(f'Found {len(self._source_list)} input images.')

    @staticmethod
    def _generate_source_list(source_path: str) -> list[str]:
        '''
        Recursively generates a list of input image paths from the specified source path.

        Args:
            source_path (str): The path where input images are located.

        Returns:
            list: A list of paths to input images.
        '''
        input_list = []

        for item in os.scandir(source_path):
            if os.path.isdir(item):
                if not item.name[0] == '.':
                    input_list.extend(ImageSource._generate_source_list(item.path))
            else:
                input_list.append(item.path)
        return input_list

    def __iter__(self) -> ImageSource:
        '''
        Returns an iterator object.

        Returns:
            ImageSource: An iterator object.
        '''
        return self

    def __next__(self):
        '''
        Returns the next image from the source path.

        Returns:
            PIL.Image: The next image from the source path.

        Raises:
            StopIteration: If there are no more images to iterate over.
        '''
        try:
            path = self._source_list.pop()
            img = Image.open(path).convert('RGB')
            return img
        except PIL.UnidentifiedImageError:
            print(f'Cannot identify image file at {path}')
            path = self._source_list.pop()
            img = Image.open(path).convert('RGB')
            return img
        except IndexError:
            raise StopIteration()


# ==================================================================================================
#                                            ARGUMENT PARSING
# ==================================================================================================

def parse_arguments() -> argparse.Namespace:
    '''
    Parses command-line arguments.

    Returns:
        argparse.Namespace: An object containing parsed arguments.
    '''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-sp', '--source_path', type=str, required=True,
                        help='Path from which images will be read.')
    parser.add_argument('-op', '--output_path', type=str, required=True,
                        help='Path to which folders with images will be saved.')
    parser.add_argument('-gs', '--grey_scale', type=str, required=False,
                        default='RGB', help='Grey scale mode. Available options: L, RGB, CMYK')
    parser.add_argument('-of', '--output_format', type=str, required=True,
                        help='Format of the output.')
    parser.add_argument('-ow', '--output_width', type=int, required=True,
                        help='Width of the output image.')
    parser.add_argument('-oh', '--output_height', type=int, required=True,
                        help='Height of the output image.')

    return parser.parse_args()


# ==================================================================================================
#                                             MAIN
# ==================================================================================================
def main(args: argparse.Namespace) -> None:
    '''
    Main function for converting images to the format chosen by the user.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    '''
    sources = ImageSource(args.source_path)
    writer = ImageWriter(args.output_path, args.grey_scale, args.output_format, args.output_width, args.output_height)
    for image in sources:
        writer.write_image(image)


if __name__ == '__main__':
    main(parse_arguments())
