`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    13:27:40 12/08/2013 
// Design Name: 
// Module Name:    gate2 
// Project Name: 
// Target Devices: 
// Tool versions: 
// Description: 
//
// Dependencies: 
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments: 
//
//////////////////////////////////////////////////////////////////////////////////
module gate2(
  input wire a,
  input wire b,
  output wire[5:0] z
);

  assign z[5] = a & b;
  assign z[4] = ~( a & b );
  assign z[3] = ( a | b );
  assign z[2] = ~( a | b );
  assign z[1] = a ^ b;
  assign z[0] = ~( a ^ b );

endmodule
