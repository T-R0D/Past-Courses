# This file is part of JST.
#
# JST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# JST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with JST.  If not, see <http://www.gnu.org/licenses/>.


class CompileError(Exception):
    def __init__(self, message, line_num, token_col, source_line):
        self.message = message
        self.line_num = line_num
        self.token_col = token_col
        self.source_line = source_line.replace('\t', ' ')

    def __str__(self):
        pointer_to_error = ""
        for i in range(self.token_col - 1):
            pointer_to_error += '-'
        pointer_to_error += '^'

        return "{}\nLine {}, Column {}\n{}\n{}".\
            format(self.message, self.line_num, self.token_col, self.source_line, pointer_to_error)

    @classmethod
    def from_tuple(cls, message, line_col_source):
        return CompileError(message, line_col_source[0], line_col_source[1], line_col_source[2])
