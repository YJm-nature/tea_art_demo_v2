"""Launch LabelImg with PyQt 5.15 numeric-overload compatibility fixes."""

from functools import wraps
from numbers import Real
import sys

from PyQt5.QtGui import QPainter


def _integer_coordinates(method, coordinate_indexes, argument_count):
    @wraps(method)
    def wrapper(self, *args):
        if len(args) == argument_count and all(
            isinstance(args[index], Real) for index in coordinate_indexes
        ):
            converted = list(args)
            for index in coordinate_indexes:
                converted[index] = int(round(float(converted[index])))
            return method(self, *converted)
        return method(self, *args)

    return wrapper


def install_pyqt_compatibility() -> None:
    # LabelImg 1.8.6 passes QPointF coordinates to integer-only overloads in
    # recent PyQt5 builds. Keep the patch local to this launcher process.
    QPainter.drawLine = _integer_coordinates(QPainter.drawLine, range(4), 4)
    QPainter.drawRect = _integer_coordinates(QPainter.drawRect, range(4), 4)
    QPainter.drawText = _integer_coordinates(QPainter.drawText, (0, 1), 3)


def main() -> int:
    install_pyqt_compatibility()
    from labelImg.labelImg import main as labelimg_main

    return labelimg_main()


if __name__ == "__main__":
    sys.exit(main())
