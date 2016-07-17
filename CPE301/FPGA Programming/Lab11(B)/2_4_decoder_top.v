`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   14:45:23 12/08/2013
// Design Name:   problem3_1
// Module Name:   C:/Documents and Settings/thenriod/Desktop/FPGA Programming/Lab11(B)/2_4_decoder_top.v
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

module 2_4_decoder_top_v;

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

	end
      
endmodule

