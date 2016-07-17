`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   14:53:55 12/08/2013
// Design Name:   decoder_2_4
// Module Name:   C:/Documents and Settings/thenriod/Desktop/FPGA Programming/Lab11(C)/deecoder2_4_top.v
// Project Name:  Lab11(C)
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: decoder_2_4
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module deecoder2_4_top_v;

	// Inputs
	reg [1:0] x;
	reg en;

	// Outputs
	wire [3:0] y;

	// Instantiate the Unit Under Test (UUT)
	decoder_2_4 uut (
		.x(x), 
		.en(en), 
		.y(y)
	);

	initial begin
		// Initialize Inputs
		x = 0;
		en = 0;

		// Wait 100 ns for global reset to finish
		#100;

		// Add stimulus here
      en = 1;
		x = 0;
		#20;
		en = 0;
		x = 0;
		#20;
		en = 0;
		x = 1;
		#20;
		en = 1;
		#20;
		en = 0;
		x = 2;
		#20;
		en = 1;
		#20;
		x = 3;
		
	end
      
endmodule

