#!/usr/bin/env python3

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

import os
import sys
import argparse
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from compiler.compiler_state import CompilerState
from exceptions.compile_error import CompileError
import mips.generation as generation


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("source", type=str, help="The C program file to compile.")
    arg_parser.add_argument("-o", "--outfile", type=str, default='STDOUT',
                            help="The name of the output file. MUST be a .asm file! (Default: STDOUT)")
    arg_parser.add_argument("-sym", "--symtable", action='store_true',
                            help="Enables the printing of symbol table in other options.")
    arg_parser.add_argument("-s", "--scandebug", type=int, choices=[0, 1, 2, 3], default=0,
                            help="The debug level for the scanner. \n 0: No debug \n 1: Tokens \n 2: Source Code \n "
                                 "3: Tokens and Source Code")
    arg_parser.add_argument("-p", "--parsedebug", type=int, choices=[0, 1, 2, 3], default=0,
                            help="The debug level for the parser. \n 0: No debug \n 1: Productions \n "
                                 " 2: Productions and Source Code \n 3: Productions, Source, Misc info")
    arg_parser.add_argument("-ast", "--astree", action='store_true',
                            help="Enables the printing of the GraphViz string after parsing.")
    arg_parser.add_argument("-tac", "--threeac", type=int, choices=[0, 1, 2], default=0,
                            help="The debug level for the 3AC. \n 0: No debug \n 1: 3AC \n "
                                 " 2: 3AC + Source")
    arg_parser.add_argument("-mips", "--mips", type=int, choices=[0, 1, 2, 3], default=0,
                            help="The debug level for the MIPS. \n 0: No debug \n 1: 3AC \n "
                                 " 2: Source \n 3: 3AC + Source")
    arg_parser.add_argument("-w", "--warnings", action='store_true',
                            help="Enables warnings being printed.")

    args = vars(arg_parser.parse_args())

    # Set Symbol Table flags
    print_table = args['symtable']

    # Set Scanner flags
    print_tokens, print_source_scanner = False, False
    if args['scandebug'] is 1:
        print_tokens = True
    elif args['scandebug'] is 2:
        print_source_scanner = True
    elif args['scandebug'] is 3:
        print_tokens = True
        print_source_scanner = True

    # Set Parser flags
    print_productions, print_source_parser, print_info = False, False, False
    if args['parsedebug'] is 1:
        print_productions = True
    elif args['parsedebug'] is 2:
        print_productions = True
        print_source_parser = True
    elif args['parsedebug'] is 3:
        print_productions = True
        print_source_parser = True
        print_info = True

    source_file = open(args['source'], 'r')
    data = source_file.read()
    compiler_state = CompilerState(print_table=print_table,
                                   print_tokens=print_tokens,
                                   print_source_scanner=print_source_scanner,
                                   print_productions=print_productions,
                                   print_source_parser=print_source_parser,
                                   print_info=print_info,
                                   print_warnings=args['warnings'])

    try:
        ast = compiler_state.parse(data)
        if args['astree']:
            print(ast.to_graph_viz_str())

        if args['threeac'] is 2:
            source_tac, tac_as_str = ast.to_3ac(include_source=True)
        else:
            source_tac, tac_as_str = ast.to_3ac()

        if args['mips'] == 1:
            generator = generation.MipsGenerator(compiler_state, inject_source=False, inject_3ac=True)
        elif args['mips'] == 2:
            generator = generation.MipsGenerator(compiler_state, inject_source=True, inject_3ac=False)
        elif args['mips'] == 3:
            generator = generation.MipsGenerator(compiler_state, inject_source=True, inject_3ac=True)
        else:
            generator = generation.MipsGenerator(compiler_state, inject_source=False, inject_3ac=False)

        generator.load(source_tac)
        generator.translate_tac_to_mips()

        if args['outfile'] != 'STDOUT':
            fout = open(args['outfile'], 'w')
            fout.write(generator.dumps())
            fout.close()
        else:
            print(generator.dumps())

    except CompileError as error:
        print(error)
    finally:
        compiler_state.teardown()

if __name__ == '__main__':
    main()
