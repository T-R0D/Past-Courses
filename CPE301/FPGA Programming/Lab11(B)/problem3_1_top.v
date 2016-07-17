`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   14:23:22 12/08/2013
// Design Name:   problem3_1
// Module Name:   C:/Documents and Settings/thenriod/Desktop/FPGA Programming/Lab11(B)/problem3_1_top.v
// Project Name:  Lab11(B)
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: problem3_1
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module problem3_1_top_v;

	// Inputs
	reg [3:0] switches;

	// Outputs
	wire [2:0] lights;

	// Instantiate the Unit Under Test (UUT)
	problem3_1 uut (
		.switches(switches), 
		.lights(lights)
	);

	initial begin
		// Initialize Inputs
		switches = 0;

		// Wait 100 ns for global reset to finish
		#100;
        
		// Add stimulus here
      switches = 0;
		#20
		switches = 2;
		#20
	   switches = 3;
		#20
		switches = 4;
	   #20
		switches = 5;
		#20
		switches = 6;
		#20
		switches = 7;
		#20
		switches = 8;
		#20
		switches = 15;
	end
      
endmodule

