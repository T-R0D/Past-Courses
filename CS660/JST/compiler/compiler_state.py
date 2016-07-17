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

import sys
from ply import lex, yacc
from parsing.jst_parser import JSTParser
from scanning.jst_lexer import JSTLexer
from symbol_table.symbol_table import SymbolTable
from loggers.logger import Logger


# A simple class to share state among objects.
#
# Used by the Lexer and Parser. Contains information and items relevant to both
# classes that does not belong exclusively in either one.
class CompilerState:
    def __init__(self,
                 print_table=False,
                 table_logfile='log_symbol_table.txt',
                 print_tokens=False, print_source_scanner=True,
                 scanner_logfile='log_scanner_tokens.txt',
                 print_productions=False, print_source_parser=False, print_info=False,
                 parser_logfile=sys.stdout,
                 print_warnings=False,
                 **kwargs):

        # Initialize variables
        self.source_code = None
        self.source_lines = None

        # Initialize table
        self.symbol_table = SymbolTable()

        # Initialize symbol table logger
        if table_logfile in {sys.stdout, sys.stderr}:
            self.symbol_table_logger = Logger(table_logfile)
        else:
            self.symbol_table_logger = Logger(open(table_logfile, 'w'))

        if print_table:
            self.symbol_table_logger.add_switch(Logger.SYMBOL_TABLE)

        # Initialize token/lexer logger
        if scanner_logfile in {sys.stdout, sys.stderr}:
            self.token_logger = Logger(scanner_logfile)
        else:
            self.token_logger = Logger(open(scanner_logfile, 'w'))

        if print_source_scanner:
            self.token_logger.add_switch(Logger.SOURCE)

        if print_tokens:
            self.token_logger.add_switch(Logger.TOKEN)

        # Initialize parser logger
        if parser_logfile in {sys.stdout, sys.stderr}:
            self.parser_logger = Logger(parser_logfile)
        else:
            self.parser_logger = Logger(open(parser_logfile, 'w'))

        if print_source_parser:
            self.parser_logger.add_switch(Logger.SOURCE)

        if print_productions:
            self.parser_logger.add_switch(Logger.PRODUCTION)

        if print_info:
            self.parser_logger.add_switch(Logger.INFO)

        # Initialize warning logger
        self.warning_logger = Logger(sys.stdout)

        if print_warnings:
            self.warning_logger.add_switch(Logger.WARNING)

        # Other stuff
        self.function_scope_entered = False

        self.insert_mode = True

        # for debugging purposes
        self.clone_symbol_table_on_next_scope_exit = False
        self.cloned_tables = []

        # Create JSTLexer and the lexer object
        self.jst_lexer = JSTLexer(self)
        self.lexer = lex.lex(module=self.jst_lexer)

        # Create JSTParser and the parser object
        self.jst_parser = JSTParser(self)
        self.parser = yacc.yacc(module=self.jst_parser, start='program')

        # we will need a reference to the symbol for the main function
        self.main_function = None

    def parse(self, source_code):
        # Lex uses 1 based indexing for line numbers.
        # We are using 0 based for source_code.
        if source_code is not None:
            self.source_code = source_code
            self.source_lines = source_code.split('\n')
        else:
            self.source_code = None
            self.source_lines = None

        # Parse using the parser object
        return self.parser.parse(input=self.source_code, lexer=self.lexer, tracking=True)

    def teardown(self):
        self.jst_lexer.teardown()
        self.jst_parser.teardown()

    def get_symbol_table_logger(self):
        return self.symbol_table_logger

    def get_token_logger(self):
        return self.token_logger

    def get_parser_logger(self):
        return self.parser_logger

    def get_warning_logger(self):
        return self.warning_logger

    def get_line_col(self, production, index):
        lexpos = production.lexpos(index)
        last_newline = self.source_code.rfind('\n', 0, lexpos)
        return production.lineno(index), max(0, lexpos - last_newline)

    def get_line_col_source(self, lineno, lexpos):
        last_newline = self.source_code.rfind('\n', 0, lexpos)
        return (
            lineno,
            max(0, lexpos - last_newline),
            self.source_lines[lineno - 1]
        )
