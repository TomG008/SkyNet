
#include "net_hls.h"
#include "hls_stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <ap_fixed.h>

using namespace std;
#define DEBUG

#define EPSILON	1e-004

extern float dw1_conv_3x3_output_stitch[3][160*2][320*2];
extern float dw1_conv_1x1_output_stitch[48][160*2][320*2];
extern float dw1_max_pool_output_stitch[48][80*2][160*2];
extern float dw2_conv_3x3_output_stitch[48][80*2][160*2];
extern float dw2_conv_1x1_output_stitch[96][80*2][160*2];
extern float dw2_max_pool_output_stitch[96][40*2][80*2];
extern float dw3_conv_3x3_output_stitch[96][40*2][80*2];
extern float dw3_conv_1x1_output_stitch[192][40*2][80*2];
extern float dw3_max_pool_output_stitch[192][20*2][40*2];
extern float dw4_conv_3x3_output_stitch[192][20*2][40*2];
extern float dw4_conv_1x1_output_stitch[384][20*2][40*2];
extern float dw5_conv_3x3_output_stitch[384][20*2][40*2];
extern float dw5_conv_1x1_output_stitch[512][20*2][40*2];

// cat dw3(ch:192 -> 768) and dw5(ch:512)
extern float dw6_conv_3x3_input_stitch[1280][20*2][40*2];
extern float dw6_conv_3x3_output_stitch[1280][20*2][40*2];
extern float dw6_conv_1x1_output_stitch[96][20*2][40*2];
extern float pw7_conv_1x1_output_stitch[10][20*2][40*2];

// reordered weights for the mysterious dw3(192->768)
extern float dw6_conv_3x3_input_reo_stitch[1280][20*2][40*2];
extern float dw6_conv_3x3_output_reo_stitch[1280][20*2][40*2];
extern float dw6_conv_1x1_output_reo_stitch[96][20*2][40*2];


/////// PL Outputs
float dw1_conv_3x3_output_PL[32][160*2][320*2];
float dw1_conv_1x1_output_PL[64][160*2][320*2];
float dw1_max_pool_output_PL[64][80*2][160*2];
float dw2_conv_3x3_output_PL[64][80*2][160*2];
float dw2_conv_1x1_output_PL[96][80*2][160*2];
float dw2_max_pool_output_PL[96][40*2][80*2];
float dw3_conv_3x3_output_PL[96][40*2][80*2];
float dw3_conv_1x1_output_PL[192][40*2][80*2];
float dw3_max_pool_output_PL[192][20*2][40*2];
float dw4_conv_3x3_output_PL[192][20*2][40*2];
float dw4_conv_1x1_output_PL[384][20*2][40*2];
float dw5_conv_3x3_output_PL[384][20*2][40*2];
float dw5_conv_1x1_output_PL[512][20*2][40*2];
float dw6_conv_3x3_input_PL[1280][20*2][40*2];
float dw6_conv_3x3_output_PL[1280][20*2][40*2];
float dw6_conv_1x1_output_PL[96][20*2][40*2];
float pw7_conv_1x1_output_PL[32][20*2][40*2];




void fill_output_dw1_conv3x3( float buf[32][44][84], int ch, int col, int row)
{
	for(int c = 0; c < 3; c++) {
		for(int h = 1; h <= 40; h++) {
			for(int w = 1; w <= 80; w++) {
				dw1_conv_3x3_output_PL[ch * 32 + c][col * 40 + h - 1][row * 80 + w - 1] = buf[c][h][w];
			}
		}
	}
}

void fill_output_dw1_conv1x1( float buf[32][44][84], int ch, int col, int row)
{
	for(int c = 0; c < 32; c++) {
		for(int h = 1; h <= 40; h++) {
			for(int w = 1; w <= 80; w++) {
				dw1_conv_1x1_output_PL[ch * 32 + c][col * 40 + h - 1][row * 80 + w - 1] = buf[c][h][w];
			}
		}
	}
}

void fill_output_dw1_pool( float buf[32][44][84], int ch, int col, int row)
{
	for(int c = 0; c < 32; c++) {
		for(int h = 1; h <= 20; h++) {
			for(int w = 1; w <= 40; w++) {
				dw1_max_pool_output_PL[ch * 32 + c][col * 20 + h - 1][row * 40 + w - 1] = buf[c][h][w];
			}
		}
	}
}


void fill_output_dw2_conv3x3( float buf[32][44][84], int ch, int col, int row)
{
	for(int c = 0; c < 32; c++) {
		for(int h = 1; h <= 40; h++) {
			for(int w = 1; w <= 80; w++) {
				dw2_conv_3x3_output_PL[ch * 32 + c][col * 40 + h - 1][row * 80 + w - 1] = buf[c][h][w];
			}
		}
	}
}

void fill_output_dw2_conv1x1( float buf[32][44][84], int ch, int col, int row)
{
	for(int c = 0; c < 32; c++) {
		for(int h = 1; h <= 40; h++) {
			for(int w = 1; w <= 80; w++) {
				dw2_conv_1x1_output_PL[ch * 32 + c][col * 40 + h - 1][row * 80 + w - 1] = buf[c][h][w];
			}
		}
	}
}

void fill_output_dw2_pool( float buf[32][44][84], int ch, int col, int row)
{
	for(int c = 0; c < 32; c++) {
		for(int h = 1; h <= 20; h++) {
			for(int w = 1; w <= 40; w++) {
				dw2_max_pool_output_PL[ch * 32 + c][col * 20 + h - 1][row * 40 + w - 1] = buf[c][h][w];
			}
		}
	}
}


void fill_output_dw3_conv3x3( float buf[32][44][84], int ch, int col, int row)
{
	for(int c = 0; c < 32; c++) {
		for(int h = 1; h <= 40; h++) {
			for(int w = 1; w <= 80; w++) {
				dw3_conv_3x3_output_PL[ch * 32 + c][col * 40 + h - 1][row * 80 + w - 1] = buf[c][h][w];
			}
		}
	}
}

void fill_output_dw3_conv1x1( float buf[32][44][84], int ch, int col, int row)
{
	for(int c = 0; c < 32; c++) {
		for(int h = 1; h <= 40; h++) {
			for(int w = 1; w <= 80; w++) {
				dw3_conv_1x1_output_PL[ch * 32 + c][col * 40 + h - 1][row * 80 + w - 1] = buf[c][h][w];
			}
		}
	}
}

void fill_output_dw3_pool( float buf[32][44][84], int ch, int col, int row)
{
	for(int c = 0; c < 32; c++) {
		for(int h = 1; h <= 20; h++) {
			for(int w = 1; w <= 40; w++) {
				dw3_max_pool_output_PL[ch * 32 + c][col * 20 + h - 1][row * 40 + w - 1] = buf[c][h][w];
			}
		}
	}
}



void fill_output_dw4_conv3x3( float buf[32][44][84], int ch, int col, int row)
{
	for(int c = 0; c < 32; c++) {
		for(int h = 1; h <= 20; h++) {
			for(int w = 1; w <= 40; w++) {
				dw4_conv_3x3_output_PL[ch * 32 + c][col * 40 + h - 1     ][row * 80 + w - 1     ] = buf[c][h     ][w     ];
				dw4_conv_3x3_output_PL[ch * 32 + c][col * 40 + h - 1 + 20][row * 80 + w - 1     ] = buf[c][h + 22][w     ];
				dw4_conv_3x3_output_PL[ch * 32 + c][col * 40 + h - 1     ][row * 80 + w - 1 + 40] = buf[c][h     ][w + 42];
				dw4_conv_3x3_output_PL[ch * 32 + c][col * 40 + h - 1 + 20][row * 80 + w - 1 + 40] = buf[c][h + 22][w + 42];
			}
		}
	}
}

void fill_output_dw4_conv1x1( float buf[32][44][84], int ch, int col, int row)
{
	for(int c = 0; c < 32; c++) {
		for(int h = 1; h <= 20; h++) {
			for(int w = 1; w <= 40; w++) {
				dw4_conv_1x1_output_PL[ch * 32 + c][col * 40 + h - 1     ][row * 80 + w - 1     ] = buf[c][h     ][w     ];
				dw4_conv_1x1_output_PL[ch * 32 + c][col * 40 + h - 1 + 20][row * 80 + w - 1     ] = buf[c][h + 22][w     ];
				dw4_conv_1x1_output_PL[ch * 32 + c][col * 40 + h - 1     ][row * 80 + w - 1 + 40] = buf[c][h     ][w + 42];
				dw4_conv_1x1_output_PL[ch * 32 + c][col * 40 + h - 1 + 20][row * 80 + w - 1 + 40] = buf[c][h + 22][w + 42];
			}
		}
	}
}


void fill_output_dw5_conv3x3( float buf[32][44][84], int ch, int col, int row)
{
	for(int c = 0; c < 32; c++) {
		for(int h = 1; h <= 20; h++) {
			for(int w = 1; w <= 40; w++) {
				dw5_conv_3x3_output_PL[ch * 32 + c][col * 40 + h - 1     ][row * 80 + w - 1     ] = buf[c][h     ][w     ];
				dw5_conv_3x3_output_PL[ch * 32 + c][col * 40 + h - 1 + 20][row * 80 + w - 1     ] = buf[c][h + 22][w     ];
				dw5_conv_3x3_output_PL[ch * 32 + c][col * 40 + h - 1     ][row * 80 + w - 1 + 40] = buf[c][h     ][w + 42];
				dw5_conv_3x3_output_PL[ch * 32 + c][col * 40 + h - 1 + 20][row * 80 + w - 1 + 40] = buf[c][h + 22][w + 42];
			}
		}
	}
}

void fill_output_dw5_conv1x1( float buf[32][44][84], int ch, int col, int row)
{
	for(int c = 0; c < 32; c++) {
		for(int h = 1; h <= 20; h++) {
			for(int w = 1; w <= 40; w++) {
				dw5_conv_1x1_output_PL[ch * 32 + c][col * 40 + h - 1     ][row * 80 + w - 1     ] = buf[c][h     ][w     ];
				dw5_conv_1x1_output_PL[ch * 32 + c][col * 40 + h - 1 + 20][row * 80 + w - 1     ] = buf[c][h + 22][w     ];
				dw5_conv_1x1_output_PL[ch * 32 + c][col * 40 + h - 1     ][row * 80 + w - 1 + 40] = buf[c][h     ][w + 42];
				dw5_conv_1x1_output_PL[ch * 32 + c][col * 40 + h - 1 + 20][row * 80 + w - 1 + 40] = buf[c][h + 22][w + 42];
			}
		}
	}
}


void fill_output_dw6_conv3x3( float buf[32][44][84], int ch, int col, int row)
{
	for(int c = 0; c < 32; c++) {
		for(int h = 1; h <= 20; h++) {
			for(int w = 1; w <= 40; w++) {
				dw6_conv_3x3_output_PL[ch * 32 + c][col * 40 + h - 1     ][row * 80 + w - 1     ] = buf[c][h     ][w     ];
				dw6_conv_3x3_output_PL[ch * 32 + c][col * 40 + h - 1 + 20][row * 80 + w - 1     ] = buf[c][h + 22][w     ];
				dw6_conv_3x3_output_PL[ch * 32 + c][col * 40 + h - 1     ][row * 80 + w - 1 + 40] = buf[c][h     ][w + 42];
				dw6_conv_3x3_output_PL[ch * 32 + c][col * 40 + h - 1 + 20][row * 80 + w - 1 + 40] = buf[c][h + 22][w + 42];
			}
		}
	}
}

void fill_output_dw6_conv1x1( float buf[32][44][84], int ch, int col, int row)
{
	for(int c = 0; c < 32; c++) {
		for(int h = 1; h <= 20; h++) {
			for(int w = 1; w <= 40; w++) {
				dw6_conv_1x1_output_PL[ch * 32 + c][col * 40 + h - 1     ][row * 80 + w - 1     ] = buf[c][h     ][w     ];
				dw6_conv_1x1_output_PL[ch * 32 + c][col * 40 + h - 1 + 20][row * 80 + w - 1     ] = buf[c][h + 22][w     ];
				dw6_conv_1x1_output_PL[ch * 32 + c][col * 40 + h - 1     ][row * 80 + w - 1 + 40] = buf[c][h     ][w + 42];
				dw6_conv_1x1_output_PL[ch * 32 + c][col * 40 + h - 1 + 20][row * 80 + w - 1 + 40] = buf[c][h + 22][w + 42];
			}
		}
	}
}


void fill_output_pw7_conv1x1( float buf[32][44][84], int ch, int col, int row)
{
	for(int c = 0; c < 32; c++) {
		for(int h = 1; h <= 20; h++) {
			for(int w = 1; w <= 40; w++) {
				pw7_conv_1x1_output_PL[ch * 32 + c][col * 40 + h - 1     ][row * 80 + w - 1     ] = buf[c][h     ][w     ];
				pw7_conv_1x1_output_PL[ch * 32 + c][col * 40 + h - 1 + 20][row * 80 + w - 1     ] = buf[c][h + 22][w     ];
				pw7_conv_1x1_output_PL[ch * 32 + c][col * 40 + h - 1     ][row * 80 + w - 1 + 40] = buf[c][h     ][w + 42];
				pw7_conv_1x1_output_PL[ch * 32 + c][col * 40 + h - 1 + 20][row * 80 + w - 1 + 40] = buf[c][h + 22][w + 42];
			}
		}
	}
}


void PL_golden_compare_dw1_conv3x3()
{
	FILE* fo;

	char* filename = "Comp_dw1_conv3x3";
	fo = fopen(filename, "w");

	for(int ch = 0; ch < 3; ch++) {
		for(int h = 0; h < 160*2; h++) {
			for(int w = 0; w < 320*2; w++) {
				if( abs(dw1_conv_3x3_output_PL[ch][h][w] - dw1_conv_3x3_output_stitch[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}
	fclose(fo);
}


void PL_golden_compare_dw1_conv1x1()
{
	FILE* fo;

	char* filename = "Comp_dw1_conv1x1";
	fo = fopen(filename, "w");

	for(int ch = 0; ch < 48; ch++) {
		for(int h = 0; h < 160*2; h++) {
			for(int w = 0; w < 320*2; w++) {
				if( abs(dw1_conv_1x1_output_PL[ch][h][w] - dw1_conv_1x1_output_stitch[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}
	fclose(fo);
}


void PL_golden_compare_dw1_pool()
{
	FILE* fo;

	char* filename = "Comp_dw1_pool";
	fo = fopen(filename, "w");

	for(int ch = 0; ch < 48; ch++) {
		for(int h = 0; h < 80*2; h++) {
			for(int w = 0; w < 160*2; w++) {
				if( abs(dw1_max_pool_output_PL[ch][h][w] - dw1_max_pool_output_stitch[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}
	fclose(fo);
}


void PL_golden_compare_dw2_conv3x3()
{
	FILE* fo;

	char* filename = "Comp_dw2_conv3x3";
	fo = fopen(filename, "w");

	for(int ch = 0; ch < 48; ch++) {
		for(int h = 0; h < 80*2; h++) {
			for(int w = 0; w < 160*2; w++) {
				if( abs(dw2_conv_3x3_output_PL[ch][h][w] - dw2_conv_3x3_output_stitch[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}
	fclose(fo);
}


void PL_golden_compare_dw2_conv1x1()
{
	FILE* fo;

	char* filename = "Comp_dw2_conv1x1";
	fo = fopen(filename, "w");

	for(int ch = 0; ch < 96; ch++) {
		for(int h = 0; h < 80*2; h++) {
			for(int w = 0; w < 160*2; w++) {
				if( abs(dw2_conv_1x1_output_PL[ch][h][w] - dw2_conv_1x1_output_stitch[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}
	fclose(fo);
}


void PL_golden_compare_dw2_pool()
{
	FILE* fo;

	char* filename = "Comp_dw2_pool";
	fo = fopen(filename, "w");

	for(int ch = 0; ch < 96; ch++) {
		for(int h = 0; h < 40*2; h++) {
			for(int w = 0; w < 80*2; w++) {
				if( abs(dw2_max_pool_output_PL[ch][h][w] - dw2_max_pool_output_stitch[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}
	fclose(fo);
}




void PL_golden_compare_dw3_conv3x3()
{
	FILE* fo;

	char* filename = "Comp_dw3_conv3x3";
	fo = fopen(filename, "w");

	for(int ch = 0; ch < 96; ch++) {
		for(int h = 0; h < 40*2; h++) {
			for(int w = 0; w < 80*2; w++) {
				if( abs(dw3_conv_3x3_output_PL[ch][h][w] - dw3_conv_3x3_output_stitch[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}
	fclose(fo);
}


void PL_golden_compare_dw3_conv1x1()
{
	FILE* fo;

	char* filename = "Comp_dw3_conv1x1";
	fo = fopen(filename, "w");

	for(int ch = 0; ch < 192; ch++) {
		for(int h = 0; h < 40*2; h++) {
			for(int w = 0; w < 80*2; w++) {
				if( abs(dw3_conv_1x1_output_PL[ch][h][w] - dw3_conv_1x1_output_stitch[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}
	fclose(fo);
}


void PL_golden_compare_dw3_pool()
{
	FILE* fo;

	char* filename = "Comp_dw3_pool";
	fo = fopen(filename, "w");

	for(int ch = 0; ch < 192; ch++) {
		for(int h = 0; h < 20*2; h++) {
			for(int w = 0; w < 40*2; w++) {
				if( abs(dw3_max_pool_output_PL[ch][h][w] - dw3_max_pool_output_stitch[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}
	fclose(fo);
}


void PL_golden_compare_dw4_conv3x3()
{
	FILE* fo;

	char* filename = "Comp_dw4_conv3x3";
	fo = fopen(filename, "w");

	for(int ch = 0; ch < 192; ch++) {
		for(int h = 0; h < 20*2; h++) {
			for(int w = 0; w < 40*2; w++) {
				if( abs(dw4_conv_3x3_output_PL[ch][h][w] - dw4_conv_3x3_output_stitch[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					if( ch == 0 ) {
						printf("[0][%d][%d] PL: %f, GC: %f\n",
								h, w, dw4_conv_3x3_output_PL[ch][h][w], dw4_conv_3x3_output_stitch[ch][h][w]);
					}
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}
	fclose(fo);
}


void PL_golden_compare_dw4_conv1x1()
{
	FILE* fo;

	char* filename = "Comp_dw4_conv1x1";
	fo = fopen(filename, "w");

	for(int ch = 0; ch < 384; ch++) {
		for(int h = 0; h < 20*2; h++) {
			for(int w = 0; w < 40*2; w++) {
				if( abs(dw4_conv_1x1_output_PL[ch][h][w] - dw4_conv_1x1_output_stitch[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}
	fclose(fo);
}



void PL_golden_compare_dw5_conv3x3()
{
	FILE* fo;

	char* filename = "Comp_dw5_conv3x3";
	fo = fopen(filename, "w");

	for(int ch = 0; ch < 384; ch++) {
		for(int h = 0; h < 20*2; h++) {
			for(int w = 0; w < 40*2; w++) {
				if( abs(dw5_conv_3x3_output_PL[ch][h][w] - dw5_conv_3x3_output_stitch[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}
	fclose(fo);
}


void PL_golden_compare_dw5_conv1x1()
{
	FILE* fo;

	char* filename = "Comp_dw5_conv1x1";
	fo = fopen(filename, "w");

	for(int ch = 0; ch < 512; ch++) {
		for(int h = 0; h < 20*2; h++) {
			for(int w = 0; w < 40*2; w++) {
				if( abs(dw5_conv_1x1_output_PL[ch][h][w] - dw5_conv_1x1_output_stitch[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}
	fclose(fo);
}


void PL_golden_compare_dw6_conv3x3()
{
	FILE* fo;

	char* filename = "Comp_dw6_conv3x3";
	fo = fopen(filename, "w");

	for(int ch = 0; ch < 1280; ch++) {
		fprintf(fo, "ch %d\n", ch);
		for(int h = 0; h < 20*2; h++) {
			for(int w = 0; w < 40*2; w++) {
				if( abs(dw6_conv_3x3_output_PL[ch][h][w] - dw6_conv_3x3_output_reo_stitch[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}
	fclose(fo);
}


void PL_golden_compare_dw6_conv1x1()
{
	FILE* fo;

	char* filename = "Comp_dw6_conv1x1";
	fo = fopen(filename, "w");

	for(int ch = 0; ch < 96; ch++) {
		for(int h = 0; h < 20*2; h++) {
			for(int w = 0; w < 40*2; w++) {

				if( abs(dw6_conv_1x1_output_PL[ch][h][w] - dw6_conv_1x1_output_stitch[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}
	fclose(fo);
}



void PL_golden_compare_pw7_conv1x1()
{
	FILE* fo;

	char* filename = "Comp_pw7_conv1x1";
	fo = fopen(filename, "w");

	for(int ch = 0; ch < 10; ch++) {
		for(int h = 0; h < 20*2; h++) {
			for(int w = 0; w < 40*2; w++) {
				if( abs(pw7_conv_1x1_output_PL[ch][h][w] - pw7_conv_1x1_output_stitch[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}
	fclose(fo);
}

