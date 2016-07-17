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


# This file was made to prevent circular dependencies, if we can do something better, let's do it

import mips.registers as mr

SPILL_MEM_LABEL = 'SPILL_MEMORY'
SPILL_MEM_SIZE = 64  # bytes
TEMPROARY_REGISTER_SET = mr.T_REGISTERS

NOT_TESTING_FUNCTIONS = False
