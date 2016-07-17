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

import argparse
import sys
import os
import json
import string

sys.path.insert(1, os.path.join(sys.path[0], '../'))

PATH_TO_THIS_SCRIPT = sys.path[0]
DEFAULT_AST_NODE_FILENAME = os.path.join(PATH_TO_THIS_SCRIPT, '../ast/ast_nodes.py')

HELP_MESSAGE = """
A configuration file should appear as (for a format example, not necessarily correct):
[
  {
    "comment": "This type of node handles all loops: for, while, and do...while.",
    "name": "IterationNode",
    "attributes": ["is_pre_test_loop"],
    "single_children": ["initialization_expression", "stop_condition_expression", "increment_expression"],
    "list_children": ["body_statments"],
    "3ac_parameters": ["a_dummy_parameter"],
    "ticket_counting": ["label_ticket_counter"]
    "python_3ac_gen_code": ""
  }
]

The only required field is "name". Although, the generated class won't be very useful without any of the children
items.
"""

FILE_EXISTS_MESSAGE = """
The specified file (given by you or the default name) already exists!
This class generator is meant to save work on boilerplate, but there will be handwritten code in the classes.
In order to prevent losing that work, this program will not attempt to generate a file.
Please copy/move the file and try again.
"""

AST_NODE_FILE_PROLOGUE = """##
# These classes are AUTO-GENERATED!
# Most of the boilerplate should be written for you, so you should carefully handwrite methods
# that are unique or need special logic for overloading.
##


class BaseAstNode(object):
    def __init__(self, **kwargs):
        pass

    def __str__(self):
        pass

"""

CLASS_TEMPLATE = """
${comment}
class ${name}(BaseAstNode):
${init_method}
${children_method}
${to_3ac_stub}
"""


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("node_configurations_file", type=str, help="")
    arg_parser.add_argument("-o", "--outfile", type=str, default=DEFAULT_AST_NODE_FILENAME,
                            help="The output file name. File must not exist yet (created by this script).")
    arg_parser.add_argument("-e", "--explain", help='Get a detailed explanation on using this program',
                            action='store_true')

    args = arg_parser.parse_args()
    args_dict = vars(args)

    if args_dict.get('explain', False):
        print(HELP_MESSAGE)
        return

    if os.path.isfile(args_dict['outfile']):
        print(FILE_EXISTS_MESSAGE)
        return

    node_definitions = []
    with open(args_dict['node_configurations_file']) as config_file:
        node_definitions = json.load(config_file)

    with open(args_dict['outfile'], 'w') as class_file:
        class_file.write(AST_NODE_FILE_PROLOGUE)

        for definition in node_definitions:
            class_definition = generate_class_definition(definition)
            class_file.write(class_definition)


def generate_class_definition(definition: dict):
    class_name = definition.get('name', None)

    if not class_name:
        raise Exception('This definition has not name for the class - unable to generate code!')

    template = string.Template(CLASS_TEMPLATE)

    comment = definition.get('comment', '')
    comment = '##\n# ' + comment + '\n##' if comment else ''

    init_method = generate_init_method(definition)
    children_method = generate_children_method(definition)
    to_3ac_method_stub = generate_to_3ac_method_stub(definition)

    return template.substitute(comment=comment, name=class_name, init_method=init_method,
                               children_method=children_method, to_3ac_stub=to_3ac_method_stub)


def generate_init_method(definition: dict):
    params = []
    params.extend(definition.get('attributes', []))
    params.extend(definition.get('single_children', []))
    params.extend(['{}=None'.format(child) for child in definition.get('list_children', [])])
    param_str = ', '.join(params) + ', ' if params else ''

    src = '    def __init__(self, {}**kwargs):\n'.format(param_str)
    src += '        super({}, self).__init__(**kwargs)\n\n'.format(definition['name'])
    for attribute in definition.get('attributes', []):
        src += '        self.{0} = {0}\n'.format(attribute)
    src += '\n'

    for child in definition.get('single_children', []):
        src += '        self.{0} = {0}\n'.format(child)
    src += '\n'

    for child in definition.get('list_children', []):
        src += '        self.{0} = {0} if {0} else []\n'.format(child)

    return src


def generate_children_method(definition: dict):
    src = '    @property\n' \
          '    def children(self):\n' \
          '        children = []\n'

    for single_child in definition.get('single_children', []):
        src += '        children.append(self.{})\n'.format(single_child)

    for list_child in definition.get('list_children', []):
        src += '        children.extend(self.{})\n'.format(list_child)

    src += '        return tuple(children)\n'

    return src

def generate_to_3ac_method_stub(definition: dict):

    parameters = definition.get("3ac_parameters", [])
    parameter_list = ', '.join(parameters) + ', ' if parameters else ''

    src =  "    def to_3ac(self, {}return_register:str=None, include_source=False):\n".format(parameter_list)
    src += "        raise NotImplementedError('Please implement the {}.to_3ac(self) method.'" \
           ".format(type(self).__name__))\n"
    return src

if __name__ == '__main__':
    main()
