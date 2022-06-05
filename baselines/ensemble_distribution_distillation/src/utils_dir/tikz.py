"""Wrapper module for tikzplotlib"""
from pathlib import Path
import tikzplotlib
from typing import Union


def save(filepath: Path,
         comment: Union[str, dict],
         figure="gcf",
         axis_width=None,
         axis_height=None,
         textsize=10.0,
         table_row_sep="\n"):

    tikzplotlib.clean_figure(fig=figure)
    output = tikzplotlib.get_tikz_code(figure=figure,
                                       filepath=filepath,
                                       axis_width=axis_width,
                                       axis_height=axis_height,
                                       textsize=textsize,
                                       table_row_sep=table_row_sep)
    output = _add_comment(output, comment)
    with open(filepath, "w") as tikz_file:
        tikz_file.write(output)


def _add_comment(output: str, comment):
    if isinstance(comment, str):
        for line in comment.split('\n'):
            output += "\n%{}".format(line)
    elif isinstance(comment, dict):
        for prop, value in comment.items():
            output += "\n%{}: {}".format(prop, value)
    return output


def test():
    output = "Here are\nsome uncommented\nstatements"
    comment = "These\nare\ncomments"
    print(_add_comment(output, comment))


if __name__ == "__main__":
    test()
