
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

FILE* fo;

/////// Golden Model Variables
extern float image[3][160][320];
extern float dw1_conv_3x3_weight[3][3][3];
extern float dw1_conv_3x3_bias[3];
extern float dw1_conv_3x3_output[3][160][320];

extern float dw1_conv_1x1_weight[48][3];
extern float dw1_conv_1x1_bias[48];
extern float dw1_conv_1x1_output[48][160][320];

extern float dw1_max_pool_output[48][80][160];

extern float dw2_conv_3x3_weight[48][3][3];
extern float dw2_conv_3x3_bias[48];
extern float dw2_conv_3x3_output[48][80][160];

extern float dw2_conv_1x1_weight[96][48];
extern float dw2_conv_1x1_bias[96];
extern float dw2_conv_1x1_output[96][80][160];

extern float dw2_max_pool_output[96][40][80];

extern float dw3_conv_3x3_weight[96][3][3];
extern float dw3_conv_3x3_bias[96];
extern float dw3_conv_3x3_output[96][40][80];

extern float dw3_conv_1x1_weight[192][96];
extern float dw3_conv_1x1_bias[192];
extern float dw3_conv_1x1_output[192][40][80];

extern float dw3_max_pool_output[192][20][40];

extern float dw4_conv_3x3_weight[192][3][3];
extern float dw4_conv_3x3_bias[192];
extern float dw4_conv_3x3_output[192][20][40];

extern float dw4_conv_1x1_weight[384][192];
extern float dw4_conv_1x1_bias[384];
extern float dw4_conv_1x1_output[384][20][40];

extern float dw5_conv_3x3_weight[384][3][3];
extern float dw5_conv_3x3_bias[384];
extern float dw5_conv_3x3_output[384][20][40];

extern float dw5_conv_1x1_weight[512][384];
extern float dw5_conv_1x1_bias[512];
extern float dw5_conv_1x1_output[512][20][40];

// cat dw3(ch:192 -> 768) and dw5(ch:512)
extern float dw6_conv_3x3_input[1280][20][40];
extern float dw6_conv_3x3_weight[1280][3][3];
extern float dw6_conv_3x3_bias[1280];
extern float dw6_conv_3x3_output[1280][20][40];

extern float dw6_conv_1x1_weight[96][1280];
extern float dw6_conv_1x1_bias[96];
extern float dw6_conv_1x1_output[96][20][40];

extern float pw7_conv_1x1_weight[10][96];
extern float pw7_conv_1x1_output[10][20][40];

// reordered weights for the mysterious dw3(192->768)
extern float dw6_conv_3x3_weight_reo[1280][3][3];
extern float dw6_conv_3x3_bias_reo[1280];
extern float dw6_conv_3x3_input_reo[1280][20][40];
extern float dw6_conv_3x3_output_reo[1280][20][40];

extern float dw6_conv_1x1_weight_reo[96][1280];
extern float dw6_conv_1x1_bias_reo[96];
extern float dw6_conv_1x1_output_reo[96][20][40];


float max_4(float a1, float a2, float a3, float a4)
{
    float tmp1, tmp2;

    if(a1 > a2) tmp1 = a1; else tmp1 = a2;
    if(a3 > a4) tmp2 = a3; else tmp2 = a4;
    if(tmp1 > tmp2) return tmp1; else return tmp2;
}



void dw1_conv_3x3(
            float input[3][160][320],
            float weight[3][3][3],
            float bias[3],
            float output[3][160][320]
            )
{
    //cout << "dw1_conv_3x3..." << endl;

    for(int co = 0; co < 3; co++) {
        for(int h = 0; h < 160; h++) {
            for(int w = 0; w < 320; w++) {
                float sum = 0;

                for(int m = 0; m < 3; m++) {
                    for(int n = 0; n < 3; n++) {

                        sum += weight[co][m][n] *
                                (( h+m-1 >= 0 && w+n-1 >= 0 && h+m-1 < 160 && w+n-1 < 320) ? input[co][h+m-1][w+n-1] : 0);


                    }
                }
                float result = sum + bias[co];
                if( result < 0 ) {
                    output[co][h][w] = 0.0f;
                }
                else if( result > 6 ) {
                    output[co][h][w] = 6.0f;
                }
                else {
                    output[co][h][w] = result;
                }
            }
        }
    }

#ifdef DEBUG
    fo = fopen("dw1_conv_3x3_output", "w");
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 160; j++) {
            for(int k = 0; k < 320; k ++) {
                fprintf(fo, "dw1_conv_3x3_output[%d][%d][%d] = %.3f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);
#endif

}


void dw1_conv_1x1(
            float input[3][160][320],
            float weight[48][3],
            float bias[48],
            float output[48][160][320]
            )
{
    //cout << "dw1_conv_1x1..." << endl;

    for(int co = 0; co < 48; co++) {
        for(int h = 0; h < 160; h++) {
            for(int w = 0; w < 320; w++) {
                float sum = 0;

                for(int ci = 0; ci < 3; ci++ ) {
                    sum += weight[co][ci] * input[ci][h][w];
                }
                float result = sum + bias[co];
                if( result < 0 ) {
                    output[co][h][w] = 0.0f;
                }
                else if( result > 6 ) {
                    output[co][h][w] = 6.0f;
                }
                else {
                    output[co][h][w] = result;
                }
            }
        }
    }

#ifdef DEBUG
    fo = fopen("dw1_conv_1x1_output", "w");
    for(int i = 0; i < 48; i++) {
        for(int j = 0; j < 160; j++) {
            for(int k = 0; k < 320; k ++) {
                fprintf(fo, "dw1_conv_1x1_output[%d][%d][%d] = %.3f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);
#endif
}


void dw1_max_pool(
                   float input[48][160][320],
                   float output[48][80][160]
                   )
{
    //cout << "dw1_max_pool..." << endl;

    for(int co = 0; co < 48; co++) {
        for(int h = 0; h < 80; h++) {
            for(int w = 0; w < 160; w++) {

                output[co][h][w] = max_4(
                                        input[co][h*2][w*2],
                                        input[co][h*2+1][w*2],
                                        input[co][h*2][w*2+1],
                                        input[co][h*2+1][w*2+1]
                                        );
            }
        }
    }

#ifdef DEBUG
    fo = fopen("dw1_max_pool_output", "w");
    for(int i = 0; i < 48; i++) {
        for(int j = 0; j < 80; j++) {
            for(int k = 0; k < 160; k ++) {
                fprintf(fo, "dw1_max_pool_output[%d][%d][%d] = %.3f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);
#endif

}


void dw2_conv_3x3(
            float input[48][80][160],
            float weight[48][3][3],
            float bias[48],
            float output[48][80][160]
            )
{
    //cout << "dw2_conv_3x3..." << endl;

    for(int co = 0; co < 48; co++) {
        for(int h = 0; h < 80; h++) {
            for(int w = 0; w < 160; w++) {
                float sum = 0;

                for(int m = 0; m < 3; m++) {
                    for(int n = 0; n < 3; n++) {
                        sum += weight[co][m][n] *
                                (( h+m-1 >= 0 && w+n-1 >= 0 && h+m-1 < 80 && w+n-1 < 160) ? input[co][h+m-1][w+n-1] : 0);
                    }
                }
                float result = sum + bias[co];
                if( result < 0 ) {
                    output[co][h][w] = 0.0f;
                }
                else if( result > 6 ) {
                    output[co][h][w] = 6.0f;
                }
                else {
                    output[co][h][w] = result;
                }
            }
        }
    }

#ifdef DEBUG
    fo = fopen("dw2_conv_3x3_output", "w");
    for(int i = 0; i < 48; i++) {
        for(int j = 0; j < 80; j++) {
            for(int k = 0; k < 160; k ++) {
                fprintf(fo, "dw2_conv_3x3_output[%d][%d][%d] = %.3f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);
#endif

}


void dw2_conv_1x1(
            float input[48][80][160],
            float weight[96][48],
            float bias[96],
            float output[96][80][160]
            )
{
    //cout << "dw2_conv_1x1..." << endl;

    for(int co = 0; co < 96; co++) {
        for(int h = 0; h < 80; h++) {
            for(int w = 0; w < 160; w++) {
                float sum = 0;

                for(int ci = 0; ci < 48; ci++ ) {
                    sum += weight[co][ci] * input[ci][h][w];
                }
                float result = sum + bias[co];
                if( result < 0 ) {
                    output[co][h][w] = 0.0f;
                }
                else if( result > 6 ) {
                    output[co][h][w] = 6.0f;
                }
                else {
                    output[co][h][w] = result;
                }
            }
        }
    }

#ifdef DEBUG
    fo = fopen("dw2_conv_1x1_output", "w");
    for(int i = 0; i < 96; i++) {
        for(int j = 0; j < 80; j++) {
            for(int k = 0; k < 160; k ++) {
                fprintf(fo, "dw2_conv_1x1_output[%d][%d][%d] = %.3f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);
#endif

}


void dw2_max_pool(
                float input[96][80][160],
                float output[96][40][80]
                )
{
    //cout << "dw2_max_pool..." << endl;

    for(int co = 0; co < 96; co++) {
        for(int h = 0; h < 40; h++) {
            for(int w = 0; w < 80; w++) {

                output[co][h][w] = max_4(
                                        input[co][h*2][w*2],
                                        input[co][h*2+1][w*2],
                                        input[co][h*2][w*2+1],
                                        input[co][h*2+1][w*2+1]
                                        );
            }
        }
    }

#ifdef DEBUG
    fo = fopen("dw2_max_pool_output", "w");
    for(int i = 0; i < 96; i++) {
        for(int j = 0; j < 40; j++) {
            for(int k = 0; k < 80; k ++) {
                fprintf(fo, "dw2_max_pool_output[%d][%d][%d] = %.3f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);
#endif

}


void dw3_conv_3x3(
            float input[96][40][80],
            float weight[96][3][3],
            float bias[96],
            float output[96][40][80]
            )
{
    //cout << "dw3_conv_3x3..." << endl;

    for(int co = 0; co < 96; co++) {
        for(int h = 0; h < 40; h++) {
            for(int w = 0; w < 80; w++) {
                float sum = 0;

                for(int m = 0; m < 3; m++) {
                    for(int n = 0; n < 3; n++) {

                        sum += weight[co][m][n] *
                                (( h+m-1 >= 0 && w+n-1 >= 0 && h+m-1 < 40 && w+n-1 < 80) ? input[co][h+m-1][w+n-1] : 0);
                    }
                }
                float result = sum + bias[co];
                if( result < 0 ) {
                    output[co][h][w] = 0.0f;
                }
                else if( result > 6 ) {
                    output[co][h][w] = 6.0f;
                }
                else {
                    output[co][h][w] = result;
                }
            }
        }
    }

#ifdef DEBUG
    fo = fopen("dw3_conv_3x3_output", "w");
    for(int i = 0; i < 96; i++) {
        for(int j = 0; j < 40; j++) {
            for(int k = 0; k < 80; k ++) {
                fprintf(fo, "dw3_conv_3x3_output[%d][%d][%d] = %.3f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);
#endif
}


void dw3_conv_1x1(
            float input[96][40][80],
            float weight[192][96],
            float bias[192],
            float output[192][40][80]
            )
{
    //cout << "dw3_conv_1x1..." << endl;

    for(int co = 0; co < 192; co++) {
        for(int h = 0; h < 40; h++) {
            for(int w = 0; w < 80; w++) {
                float sum = 0;

                for(int ci = 0; ci < 96; ci++ ) {
                    sum += weight[co][ci] * input[ci][h][w];
                }
                float result = sum + bias[co];
                if( result < 0 ) {
                    output[co][h][w] = 0.0f;
                }
                else if( result > 6 ) {
                    output[co][h][w] = 6.0f;
                }
                else {
                    output[co][h][w] = result;
                }
            }
        }
    }
#ifdef DEBUG
    fo = fopen("dw3_conv_1x1_output", "w");
    for(int i = 0; i < 192; i++) {
        for(int j = 0; j < 40; j++) {
            for(int k = 0; k < 80; k ++) {
                fprintf(fo, "dw3_conv_1x1_output[%d][%d][%d] = %.5f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);
#endif
}


void dw3_max_pool(
                float input[192][40][80],
                float output[192][20][40]
                )
{
    //cout << "dw3_max_pool..." << endl;

    for(int co = 0; co < 192; co++) {
        for(int h = 0; h < 20; h++) {
            for(int w = 0; w < 40; w++) {

                output[co][h][w] = max_4(
                                        input[co][h*2][w*2],
                                        input[co][h*2+1][w*2],
                                        input[co][h*2][w*2+1],
                                        input[co][h*2+1][w*2+1]
                                        );
            }
        }
    }
#ifdef DEBUG
    fo = fopen("dw3_max_pool_output", "w");
    for(int i = 0; i < 192; i++) {
        for(int j = 0; j < 20; j++) {
            for(int k = 0; k < 40; k ++) {
                fprintf(fo, "dw3_max_pool_output[%d][%d][%d] = %.3f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);
#endif
}



void dw4_conv_3x3(
            float input[192][20][40],
            float weight[192][3][3],
            float bias[192],
            float output[192][20][40]
            )
{
    //cout << "dw4_conv_3x3..." << endl;

    for(int co = 0; co < 192; co++) {
        for(int h = 0; h < 20; h++) {
            for(int w = 0; w < 40; w++) {
                float sum = 0;

                for(int m = 0; m < 3; m++) {
                    for(int n = 0; n < 3; n++) {

                        sum += weight[co][m][n] *
                                (( h+m-1 >= 0 && w+n-1 >= 0 && h+m-1 < 20 && w+n-1 < 40) ? input[co][h+m-1][w+n-1] : 0);
                    }
                }
                float result = sum + bias[co];
                if( result < 0 ) {
                    output[co][h][w] = 0.0f;
                }
                else if( result > 6 ) {
                    output[co][h][w] = 6.0f;
                }
                else {
                    output[co][h][w] = result;
                }
            }
        }
    }
#ifdef DEBUG
    fo = fopen("dw4_conv_3x3_output", "w");
    for(int i = 0; i < 192; i++) {
        for(int j = 0; j < 20; j++) {
            for(int k = 0; k < 40; k ++) {
                fprintf(fo, "dw4_conv_3x3_output[%d][%d][%d] = %.3f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);
#endif
}


void dw4_conv_1x1(
            float input[192][20][40],
            float weight[384][192],
            float bias[384],
            float output[384][20][40]
            )
{
    //cout << "dw4_conv_1x1..." << endl;

    for(int co = 0; co < 384; co++) {
        for(int h = 0; h < 20; h++) {
            for(int w = 0; w < 40; w++) {
                float sum = 0;

                for(int ci = 0; ci < 192; ci++ ) {
                    sum += weight[co][ci] * input[ci][h][w];
                }
                float result = sum + bias[co];
                if( result < 0 ) {
                    output[co][h][w] = 0.0f;
                }
                else if( result > 6 ) {
                    output[co][h][w] = 6.0f;
                }
                else {
                    output[co][h][w] = result;
                }
            }
        }
    }
#ifdef DEBUG
    fo = fopen("dw4_conv_1x1_output", "w");
    for(int i = 0; i < 384; i++) {
        for(int j = 0; j < 20; j++) {
            for(int k = 0; k < 40; k ++) {
                fprintf(fo, "dw4_conv_1x1_output[%d][%d][%d] = %.3f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);
#endif
}


void dw5_conv_3x3(
            float input[384][20][40],
            float weight[384][3][3],
            float bias[384],
            float output[384][20][40]
            )
{
    //cout << "dw5_conv_3x3..." << endl;

    for(int co = 0; co < 384; co++) {
        for(int h = 0; h < 20; h++) {
            for(int w = 0; w < 40; w++) {
                float sum = 0;

                for(int m = 0; m < 3; m++) {
                    for(int n = 0; n < 3; n++) {

                        sum += weight[co][m][n] *
                                (( h+m-1 >= 0 && w+n-1 >= 0 && h+m-1 < 20 && w+n-1 < 40) ? input[co][h+m-1][w+n-1] : 0);
                    }
                }
                float result = sum + bias[co];
                if( result < 0 ) {
                    output[co][h][w] = 0.0f;
                }
                else if( result > 6 ) {
                    output[co][h][w] = 6.0f;
                }
                else {
                    output[co][h][w] = result;
                }
            }
        }
    }
#ifdef DEBUG
    fo = fopen("dw5_conv_3x3_output", "w");
    for(int i = 0; i < 384; i++) {
        for(int j = 0; j < 20; j++) {
            for(int k = 0; k < 40; k ++) {
                fprintf(fo, "dw5_conv_3x3_output[%d][%d][%d] = %.3f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);
#endif
}



void dw5_conv_1x1(
            float input[384][20][40],
            float weight[512][384],
            float bias[512],
            float output[512][20][40]
            )
{
    //cout << "dw5_conv_1x1..." << endl;

    for(int co = 0; co < 512; co++) {
        for(int h = 0; h < 20; h++) {
            for(int w = 0; w < 40; w++) {
                float sum = 0;

                for(int ci = 0; ci < 384; ci++ ) {
                    sum += weight[co][ci] * input[ci][h][w];
                }
                float result = sum + bias[co];
                if( result < 0 ) {
                    output[co][h][w] = 0.0f;
                }
                else if( result > 6 ) {
                    output[co][h][w] = 6.0f;
                }
                else {
                    output[co][h][w] = result;
                }
            }
        }
    }
#ifdef DEBUG
    fo = fopen("dw5_conv_1x1_output", "w");
    for(int i = 0; i < 512; i++) {
        for(int j = 0; j < 20; j++) {
            for(int k = 0; k < 40; k ++) {
                fprintf(fo, "dw5_conv_1x1_output[%d][%d][%d] = %.3f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);
#endif
}



void concat(
            float dw3_conv_1x1_output[192][40][80],
            float dw5_conv_1x1_output[512][20][40],
            float dw6_conv_3x3_input[1280][20][40]
            )
{

    //cout << "cat dw3(ch:192 -> 768) and dw5(ch:512)..." << endl;

    // First reorder dw3_conv_1x1_output: ch192 -> ch768
    for(int co = 0; co < 192; co++) {
        for(int h = 0; h < 40; h+=4) {
            for(int w = 0; w < 80; w+=4) {
                dw6_conv_3x3_input[co][h/2][w/2]       = dw3_conv_1x1_output[co][h][w];
                dw6_conv_3x3_input[co][h/2+1][w/2]     = dw3_conv_1x1_output[co][h+2][w];
                dw6_conv_3x3_input[co][h/2][w/2+1]     = dw3_conv_1x1_output[co][h][w+2];
                dw6_conv_3x3_input[co][h/2+1][w/2+1]   = dw3_conv_1x1_output[co][h+2][w+2];

                dw6_conv_3x3_input[co+192*1][h/2][w/2]     = dw3_conv_1x1_output[co][h][w+1];
                dw6_conv_3x3_input[co+192*1][h/2+1][w/2]   = dw3_conv_1x1_output[co][h+2][w+1];
                dw6_conv_3x3_input[co+192*1][h/2][w/2+1]   = dw3_conv_1x1_output[co][h][w+3];
                dw6_conv_3x3_input[co+192*1][h/2+1][w/2+1] = dw3_conv_1x1_output[co][h+2][w+3];

                dw6_conv_3x3_input[co+192*2][h/2][w/2]     = dw3_conv_1x1_output[co][h+1][w];
                dw6_conv_3x3_input[co+192*2][h/2+1][w/2]   = dw3_conv_1x1_output[co][h+3][w];
                dw6_conv_3x3_input[co+192*2][h/2][w/2+1]   = dw3_conv_1x1_output[co][h+1][w+2];
                dw6_conv_3x3_input[co+192*2][h/2+1][w/2+1] = dw3_conv_1x1_output[co][h+3][w+2];

                dw6_conv_3x3_input[co+192*3][h/2][w/2]     = dw3_conv_1x1_output[co][h+1][w+1];
                dw6_conv_3x3_input[co+192*3][h/2+1][w/2]   = dw3_conv_1x1_output[co][h+3][w+1];
                dw6_conv_3x3_input[co+192*3][h/2][w/2+1]   = dw3_conv_1x1_output[co][h+1][w+3];
                dw6_conv_3x3_input[co+192*3][h/2+1][w/2+1] = dw3_conv_1x1_output[co][h+3][w+3];
            }
        }
    }

    for(int co = 0; co < 512; co++) {
        for(int h = 0; h < 20; h++) {
            for(int w = 0; w < 40; w++) {
                dw6_conv_3x3_input[co + 768][h][w] = dw5_conv_1x1_output[co][h][w];
            }
        }
    }

#ifdef DEBUG
    fo = fopen("dw6_conv_3x3_input", "w");
    for(int i = 0; i < 1280; i++) {
        for(int j = 0; j < 20; j++) {
            for(int k = 0; k < 40; k ++) {
                fprintf(fo, "dw6_conv_3x3_input[%d][%d][%d] = %.3f\n", i, j, k, dw6_conv_3x3_input[i][j][k]);
            }
        }
    }
    fclose(fo);


    fo = fopen("dw3_conv_3x3_output_reorg_before", "w");
    for(int i = 0; i < 192; i++) {
        for(int j = 0; j < 40; j++) {
            for(int k = 0; k < 80; k ++) {
                fprintf(fo, "dw3_conv_3x3_output_reorg_before[%d][%d][%d] = %.3f\n", i, j, k, dw3_conv_1x1_output[i][j][k]);
            }
        }
    }
    fclose(fo);

    fo = fopen("dw3_conv_3x3_output_reorg_after", "w");
    for(int i = 0; i < 768; i++) {
        for(int j = 0; j < 20; j++) {
            for(int k = 0; k < 40; k ++) {
                fprintf(fo, "dw3_conv_3x3_output_reorg_after[%d][%d][%d] = %.3f\n", i, j, k, dw6_conv_3x3_input[i][j][k]);
            }
        }
    }
    fclose(fo);


#endif

}




void concat_reo(
            float dw3_conv_1x1_output[192][40][80],
            float dw5_conv_1x1_output[512][20][40],
            float dw6_conv_3x3_input[1280][20][40]
            )
{

    //cout << "cat dw3(ch:192 -> 768) and dw5(ch:512)..." << endl;

    // First reorder dw3_conv_1x1_output: ch192 -> ch768
    for(int co = 0; co < 192; co++) {
        for(int h = 0; h < 40; h+=4) {
            for(int w = 0; w < 80; w+=4) {
                dw6_conv_3x3_input[co*4][h/2  ][w/2  ]     = dw3_conv_1x1_output[co][h  ][w  ];
                dw6_conv_3x3_input[co*4][h/2+1][w/2  ]     = dw3_conv_1x1_output[co][h+2][w  ];
                dw6_conv_3x3_input[co*4][h/2  ][w/2+1]     = dw3_conv_1x1_output[co][h  ][w+2];
                dw6_conv_3x3_input[co*4][h/2+1][w/2+1]     = dw3_conv_1x1_output[co][h+2][w+2];

                dw6_conv_3x3_input[co*4+1][h/2  ][w/2  ]   = dw3_conv_1x1_output[co][h  ][w+1];
                dw6_conv_3x3_input[co*4+1][h/2+1][w/2  ]   = dw3_conv_1x1_output[co][h+2][w+1];
                dw6_conv_3x3_input[co*4+1][h/2  ][w/2+1]   = dw3_conv_1x1_output[co][h  ][w+3];
                dw6_conv_3x3_input[co*4+1][h/2+1][w/2+1]   = dw3_conv_1x1_output[co][h+2][w+3];

                dw6_conv_3x3_input[co*4+2][h/2  ][w/2  ]   = dw3_conv_1x1_output[co][h+1][w  ];
                dw6_conv_3x3_input[co*4+2][h/2+1][w/2  ]   = dw3_conv_1x1_output[co][h+3][w  ];
                dw6_conv_3x3_input[co*4+2][h/2  ][w/2+1]   = dw3_conv_1x1_output[co][h+1][w+2];
                dw6_conv_3x3_input[co*4+2][h/2+1][w/2+1]   = dw3_conv_1x1_output[co][h+3][w+2];

                dw6_conv_3x3_input[co*4+3][h/2  ][w/2  ]   = dw3_conv_1x1_output[co][h+1][w+1];
                dw6_conv_3x3_input[co*4+3][h/2+1][w/2  ]   = dw3_conv_1x1_output[co][h+3][w+1];
                dw6_conv_3x3_input[co*4+3][h/2  ][w/2+1]   = dw3_conv_1x1_output[co][h+1][w+3];
                dw6_conv_3x3_input[co*4+3][h/2+1][w/2+1]   = dw3_conv_1x1_output[co][h+3][w+3];
            }
        }
    }

    for(int co = 0; co < 512; co++) {
        for(int h = 0; h < 20; h++) {
            for(int w = 0; w < 40; w++) {
                dw6_conv_3x3_input[co + 768][h][w] = dw5_conv_1x1_output[co][h][w];
            }
        }
    }

#ifdef DEBUG
    fo = fopen("dw6_conv_3x3_input_reo", "w");
    for(int i = 0; i < 1280; i++) {
        for(int j = 0; j < 20; j++) {
            for(int k = 0; k < 40; k ++) {
                fprintf(fo, "dw6_conv_3x3_input_reo[%d][%d][%d] = %.3f\n", i, j, k, dw6_conv_3x3_input[i][j][k]);
            }
        }
    }
    fclose(fo);

#endif

}



void dw6_conv_3x3(
            float input[1280][20][40],
            float weight[1280][3][3],
            float bias[1280],
            float output[1280][20][40]
            )
{
    //cout << "dw6_conv_3x3..." << endl;

    for(int co = 0; co < 1280; co++) {
        for(int h = 0; h < 20; h++) {
            for(int w = 0; w < 40; w++) {
                float sum = 0;

                for(int m = 0; m < 3; m++) {
                    for(int n = 0; n < 3; n++) {

                        sum += weight[co][m][n] *
                                (( h+m-1 >= 0 && w+n-1 >= 0 && h+m-1 < 20 && w+n-1 < 40) ? input[co][h+m-1][w+n-1] : 0);
                    }
                }
                float result = sum + bias[co];
                if( result < 0 ) {
                    output[co][h][w] = 0.0f;
                }
                else if( result > 6 ) {
                    output[co][h][w] = 6.0f;
                }
                else {
                    output[co][h][w] = result;
                }
            }
        }
    }
#ifdef DEBUG
    fo = fopen("dw6_conv_3x3_output", "w");
    for(int i = 0; i < 1280; i++) {
        for(int j = 0; j < 20; j++) {
            for(int k = 0; k < 40; k ++) {
                fprintf(fo, "dw6_conv_3x3_output[%d][%d][%d] = %.3f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);
#endif
}



void dw6_conv_3x3_reo(
            float input[1280][20][40],
            float weight[1280][3][3],
            float bias[1280],
            float output[1280][20][40]
            )
{
    //cout << "dw6_conv_3x3..." << endl;

    for(int co = 0; co < 1280; co++) {
        for(int h = 0; h < 20; h++) {
            for(int w = 0; w < 40; w++) {
                float sum = 0;

                for(int m = 0; m < 3; m++) {
                    for(int n = 0; n < 3; n++) {

                        sum += weight[co][m][n] *
                                (( h+m-1 >= 0 && w+n-1 >= 0 && h+m-1 < 20 && w+n-1 < 40) ? input[co][h+m-1][w+n-1] : 0);
                    }
                }
                float result = sum + bias[co];
                if( result < 0 ) {
                    output[co][h][w] = 0.0f;
                }
                else if( result > 6 ) {
                    output[co][h][w] = 6.0f;
                }
                else {
                    output[co][h][w] = result;
                }
            }
        }
    }
#ifdef DEBUG
    fo = fopen("dw6_conv_3x3_output_reo", "w");
    for(int i = 0; i < 1280; i++) {
        for(int j = 0; j < 20; j++) {
            for(int k = 0; k < 40; k ++) {
                fprintf(fo, "dw6_conv_3x3_output[%d][%d][%d] = %.3f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);
#endif
}



void dw6_conv_1x1(
            float input[1280][20][40],
            float weight[96][1280],
            float bias[96],
            float output[96][20][40]
            )
{
    //cout << "dw6_conv_1x1..." << endl;

    for(int co = 0; co < 96; co++) {
        for(int h = 0; h < 20; h++) {
            for(int w = 0; w < 40; w++) {
                float sum = 0;

                for(int ci = 0; ci < 1280; ci++ ) {
                    sum += weight[co][ci] * input[ci][h][w];
                }
                float result = sum + bias[co];
                if( result < 0 ) {
                    output[co][h][w] = 0.0f;
                }
                else if( result > 6 ) {
                    output[co][h][w] = 6.0f;
                }
                else {
                    output[co][h][w] = result;
                }
            }
        }
    }
#ifdef DEBUG
    fo = fopen("dw6_conv_1x1_output", "w");
    for(int i = 0; i < 96; i++) {
        for(int j = 0; j < 20; j++) {
            for(int k = 0; k < 40; k ++) {
                fprintf(fo, "dw6_conv_1x1_output[%d][%d][%d] = %.3f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);
#endif
}




void dw6_conv_1x1_reo(
            float input[1280][20][40],
            float weight[96][1280],
            float bias[96],
            float output[96][20][40]
            )
{
    //cout << "dw6_conv_1x1..." << endl;

    for(int co = 0; co < 96; co++) {
        for(int h = 0; h < 20; h++) {
            for(int w = 0; w < 40; w++) {
                float sum = 0;

                for(int ci = 0; ci < 1280; ci++ ) {
                    sum += weight[co][ci] * input[ci][h][w];
                }
                float result = sum + bias[co];
                if( result < 0 ) {
                    output[co][h][w] = 0.0f;
                }
                else if( result > 6 ) {
                    output[co][h][w] = 6.0f;
                }
                else {
                    output[co][h][w] = result;
                }
            }
        }
    }
#ifdef DEBUG
    fo = fopen("dw6_conv_1x1_output_reo", "w");
    for(int i = 0; i < 96; i++) {
        for(int j = 0; j < 20; j++) {
            for(int k = 0; k < 40; k ++) {
                fprintf(fo, "dw6_conv_1x1_output[%d][%d][%d] = %.3f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);
#endif
}




void pw7_conv_1x1(
            float input[96][20][40],
            float weight[10][96],
            float output[10][20][40]
            )
{
    //cout << "pw7_conv_1x1..." << endl;

    for(int co = 0; co < 10; co++) {
        for(int h = 0; h < 20; h++) {
            for(int w = 0; w < 40; w++) {
                float sum = 0;

                for(int ci = 0; ci < 96; ci++ ) {
                    sum += weight[co][ci] * input[ci][h][w];
                }
                output[co][h][w] = sum;
            }
        }
    }

#ifdef DEBUG
    fo = fopen("pw7_conv_1x1_output", "w");
    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 20; j++) {
            for(int k = 0; k < 40; k ++) {
                fprintf(fo, "pw7_conv_1x1_output[%d][%d][%d] = %.3f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);
#endif
}



void compute_bounding_box( float pw7_conv_1x1_out[10][20][40], int predict_box[5])
{
    int batch_size = 1;
    int num_anchors = 2;
    int h = 20;
    int w = 40;

    float box[4] = {1.4940052559648322, 2.3598481287086823, 4.0113013115312155, 5.760873975661669};

    float conf_thresh = 0.0;
    int conf_j = 0;
    int conf_m = 0;
    int conf_n = 0;


    float output[10][20][40];
    for(int i = 0; i < 10; i++) {
    	for(int j = 0; j < 20; j++) {
    		for(int k = 0; k < 40; k++) {
    			output[i][j][k] = pw7_conv_1x1_out[i][j][k];
    		}
    	}
    }



    //preprocessing anchor boxes xs and ys
    for(int j = 0;j < num_anchors;j++){
        for(int m = 0;m < h;m++){
            for(int n = 0;n < w;n++){
                output[j*5][m][n] = 1/(1+exp(-output[j*5][m][n]))+n;
            }
        }
    }

    for(int j = 0;j < num_anchors;j++){
        for(int m = 0;m < h;m++){
            for(int n = 0;n < w;n++){
                output[j*5+1][m][n] = 1/(1+exp(-output[j*5+1][m][n]))+m;
            }
        }
    }
    //preprocessing anchor boxes ws and hs
    for(int j = 0;j < num_anchors;j++){
        for(int m = 0;m < h;m++){
            for(int n = 0;n < w;n++){
                output[j*5+2][m][n] = exp(output[j*5+2][m][n])*box[j*2];
            }
        }
    }
    for(int j = 0;j < num_anchors;j++){
        for(int m = 0;m < h;m++){
            for(int n = 0;n < w;n++){
                output[j*5+3][m][n] = exp(output[j*5+3][m][n])*box[j*2+1];
            }
        }
    }
    //preprocessing anchor boxes det_conf
    for(int j = 0;j < num_anchors;j++){
        for(int m = 0;m < h;m++){
            for(int n = 0;n < w;n++){
                output[j*5+4][m][n] = 1/(1+exp(-output[j*5+4][m][n]));
            }
        }
    }

    //find the maximum num
    for(int j = 0;j < num_anchors;j++){
        for(int m = 0;m < h;m++){
            for(int n = 0;n < w;n++){
                if(output[j*5+4][m][n] > conf_thresh){
                    conf_thresh = output[j*5+4][m][n];
                    conf_j = j;
                    conf_m = m;
                    conf_n = n;
                }
            }
        }
    }

    //calculate the output

    float bbox[5] = {0.0};
    bbox[0] = output[conf_j*5+0][conf_m][conf_n]/w;
    bbox[1] = output[conf_j*5+1][conf_m][conf_n]/h;
    bbox[2] = output[conf_j*5+2][conf_m][conf_n]/w;
    bbox[3] = output[conf_j*5+3][conf_m][conf_n]/h;
    bbox[4] = output[conf_j*5+4][conf_m][conf_n];

     for(int i = 0; i < 5; i++){
         printf("%f ", bbox[i]);
     }
     printf("\n");

    predict_box[0] = (unsigned int)(((bbox[0] - bbox[2]/2.0)) * 640);
    predict_box[1] = (unsigned int)(((bbox[1] - bbox[3]/2.0)) * 360);
    predict_box[2] = (unsigned int)(((bbox[0] + bbox[2]/2.0)) * 640);
    predict_box[3] = (unsigned int)(((bbox[1] + bbox[3]/2.0)) * 360);

}





int golden_model()
{

    // reorder dw6_conv_3x3_weight and dw6_conv_3x3_bias
    for(int ch = 0; ch < 768; ch += 4) {
        dw6_conv_3x3_bias_reo[ch] = dw6_conv_3x3_bias[ ch / 4];
        dw6_conv_3x3_bias_reo[ch + 1] = dw6_conv_3x3_bias[ ch / 4 + 192];
        dw6_conv_3x3_bias_reo[ch + 2] = dw6_conv_3x3_bias[ ch / 4 + 192 * 2];
        dw6_conv_3x3_bias_reo[ch + 3] = dw6_conv_3x3_bias[ ch / 4 + 192 * 3];

        for(int m = 0; m < 3; m++) {
            for(int n = 0; n < 3; n++) {
                dw6_conv_3x3_weight_reo[ch][m][n] = dw6_conv_3x3_weight[ ch / 4 ][m][n];
                dw6_conv_3x3_weight_reo[ch + 1][m][n] = dw6_conv_3x3_weight[ ch / 4 + 192][m][n];
                dw6_conv_3x3_weight_reo[ch + 2][m][n] = dw6_conv_3x3_weight[ ch / 4 + 192 * 2][m][n];
                dw6_conv_3x3_weight_reo[ch + 3][m][n] = dw6_conv_3x3_weight[ ch / 4 + 192 * 3][m][n];
            }
        }

    }
    for(int ch = 768; ch < 1280; ch++) {
        dw6_conv_3x3_bias_reo[ch] = dw6_conv_3x3_bias[ch];

        for(int m = 0; m < 3; m++) {
            for(int n = 0; n < 3; n++) {
                dw6_conv_3x3_weight_reo[ch][m][n] = dw6_conv_3x3_weight[ch][m][n];
            }
        }
    }


    // reorder dw6_conv_1x1_weight and dw6_conv_1x1_bias

    for(int ci = 0; ci < 768; ci += 4) {
        for(int co = 0; co < 96; co++) {
            dw6_conv_1x1_weight_reo[co][ci] = dw6_conv_1x1_weight[co][ ci / 4 ];
            dw6_conv_1x1_weight_reo[co][ci + 1] = dw6_conv_1x1_weight[co][ ci / 4 + 192];
            dw6_conv_1x1_weight_reo[co][ci + 2] = dw6_conv_1x1_weight[co][ ci / 4 + 192 * 2];
            dw6_conv_1x1_weight_reo[co][ci + 3] = dw6_conv_1x1_weight[co][ ci / 4 + 192 * 3];
        }
    }

    for(int ci = 768; ci < 1280; ci++) {
        for(int co = 0; co < 96; co++) {
            dw6_conv_1x1_weight_reo[co][ci] = dw6_conv_1x1_weight[co][ci];
        }
    }

    for(int co = 0; co < 96; co++) {
        dw6_conv_1x1_bias_reo[co] = dw6_conv_1x1_bias[co];
    }


	dw1_conv_3x3(image, dw1_conv_3x3_weight, dw1_conv_3x3_bias, dw1_conv_3x3_output);
	dw1_conv_1x1(dw1_conv_3x3_output, dw1_conv_1x1_weight, dw1_conv_1x1_bias, dw1_conv_1x1_output);
	dw1_max_pool(dw1_conv_1x1_output, dw1_max_pool_output);

	dw2_conv_3x3(dw1_max_pool_output, dw2_conv_3x3_weight, dw2_conv_3x3_bias, dw2_conv_3x3_output);
	dw2_conv_1x1(dw2_conv_3x3_output, dw2_conv_1x1_weight, dw2_conv_1x1_bias, dw2_conv_1x1_output);
	dw2_max_pool(dw2_conv_1x1_output, dw2_max_pool_output);

	dw3_conv_3x3(dw2_max_pool_output, dw3_conv_3x3_weight, dw3_conv_3x3_bias, dw3_conv_3x3_output);
	dw3_conv_1x1(dw3_conv_3x3_output, dw3_conv_1x1_weight, dw3_conv_1x1_bias, dw3_conv_1x1_output);
	dw3_max_pool(dw3_conv_1x1_output, dw3_max_pool_output);

	dw4_conv_3x3(dw3_max_pool_output, dw4_conv_3x3_weight, dw4_conv_3x3_bias, dw4_conv_3x3_output);
	dw4_conv_1x1(dw4_conv_3x3_output, dw4_conv_1x1_weight, dw4_conv_1x1_bias, dw4_conv_1x1_output);

	dw5_conv_3x3(dw4_conv_1x1_output, dw5_conv_3x3_weight, dw5_conv_3x3_bias, dw5_conv_3x3_output);
	dw5_conv_1x1(dw5_conv_3x3_output, dw5_conv_1x1_weight, dw5_conv_1x1_bias, dw5_conv_1x1_output);

	concat(dw3_conv_1x1_output, dw5_conv_1x1_output, dw6_conv_3x3_input);
	concat_reo(dw3_conv_1x1_output, dw5_conv_1x1_output, dw6_conv_3x3_input_reo);

	dw6_conv_3x3(dw6_conv_3x3_input,  dw6_conv_3x3_weight, dw6_conv_3x3_bias, dw6_conv_3x3_output);
	dw6_conv_3x3_reo(dw6_conv_3x3_input_reo,  dw6_conv_3x3_weight_reo, dw6_conv_3x3_bias_reo, dw6_conv_3x3_output_reo);

	dw6_conv_1x1(dw6_conv_3x3_output, dw6_conv_1x1_weight, dw6_conv_1x1_bias, dw6_conv_1x1_output);
	dw6_conv_1x1_reo(dw6_conv_3x3_output_reo, dw6_conv_1x1_weight_reo, dw6_conv_1x1_bias_reo, dw6_conv_1x1_output_reo);

	pw7_conv_1x1(dw6_conv_1x1_output, pw7_conv_1x1_weight, pw7_conv_1x1_output);

	int predict_box[5];
	compute_bounding_box(pw7_conv_1x1_output, predict_box);
    printf("%d %d %d %d\n", predict_box[0], predict_box[1], predict_box[2], predict_box[3]);

    return 0;
}



