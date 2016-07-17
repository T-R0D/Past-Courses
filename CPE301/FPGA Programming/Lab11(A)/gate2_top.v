`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   13:36:08 12/08/2013
// Design Name:   gate2
// Module Name:   C:/Documents and Settings/thenriod/Desktop/FPGA Programming/Lab11(A)/gate2_top.v
// Project Name:  Lab11(A)
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: gate2
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module gate2_top_v;

	// Inputs
	reg a;
	reg b;

	// Outputs
	wire [5:0] z;

	// Instantiate the Unit Under Test (UUT)
	gate2 uut (
		.a(a), 
		.b(b), 
		.z(z)
	);

	initial begin
		// Initialize Inputs
		a = 0;
		b = 0;

		// Wait 100 ns for global reset to finish
		#100;

		// Add stimulus here
      a = 1;
		#20;
		a = 0;
		b = 1;
		#20;
		a = 1;
	end
      
endmodule

