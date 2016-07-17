`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    14:15:07 12/08/2013 
// Design Name: 
// Module Name:    problem3_1 
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
module problem3_1
(
  input wire[3:0] switches,
  output wire[2:0] lights
);

  assign lights[2] = ~( &switches );
  assign lights[1] = ~( |switches );
  assign lights[0] = ~( ^switches );

endmodule
