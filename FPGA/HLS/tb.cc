
#include "net_hls.h"
#include "hls_stream.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <ap_fixed.h>


float image[3][160][320];
float dw1_conv_3x3_weight[3][3][3];
float dw1_conv_3x3_bias[3];
float dw1_conv_3x3_output[3][160][320];

float dw1_conv_1x1_weight[48][3];
float dw1_conv_1x1_bias[48];
float dw1_conv_1x1_output[48][160][320];

float dw1_max_pool_output[48][80][160];

float dw2_conv_3x3_weight[48][3][3];
float dw2_conv_3x3_bias[48];
float dw2_conv_3x3_output[48][80][160];

float dw2_conv_1x1_weight[96][48];
float dw2_conv_1x1_bias[96];
float dw2_conv_1x1_output[96][80][160];

float dw2_max_pool_output[96][40][80];

float dw3_conv_3x3_weight[96][3][3];
float dw3_conv_3x3_bias[96];
float dw3_conv_3x3_output[96][40][80];

float dw3_conv_1x1_weight[192][96];
float dw3_conv_1x1_bias[192];
float dw3_conv_1x1_output[192][40][80];

float dw3_max_pool_output[192][20][40];

float dw4_conv_3x3_weight[192][3][3];
float dw4_conv_3x3_bias[192];
float dw4_conv_3x3_output[192][20][40];

float dw4_conv_1x1_weight[384][192];
float dw4_conv_1x1_bias[384];
float dw4_conv_1x1_output[384][20][40];

float dw5_conv_3x3_weight[384][3][3];
float dw5_conv_3x3_bias[384];
float dw5_conv_3x3_output[384][20][40];

float dw5_conv_1x1_weight[512][384];
float dw5_conv_1x1_bias[512];
float dw5_conv_1x1_output[512][20][40];

// cat dw3(ch:192 -> 768) and dw5(ch:512)
float dw6_conv_3x3_input[1280][20][40];
float dw6_conv_3x3_weight[1280][3][3];
float dw6_conv_3x3_bias[1280];
float dw6_conv_3x3_output[1280][20][40];

float dw6_conv_1x1_weight[96][1280];
float dw6_conv_1x1_bias[96];
float dw6_conv_1x1_output[96][20][40];

float pw7_conv_1x1_weight[10][96];
float pw7_conv_1x1_output[10][20][40];


// reordered weights for the mysterious dw3(192->768)
float dw6_conv_3x3_weight_reo[1280][3][3];
float dw6_conv_3x3_bias_reo[1280];
float dw6_conv_3x3_input_reo[1280][20][40];
float dw6_conv_3x3_output_reo[1280][20][40];

float dw6_conv_1x1_weight_reo[96][1280];
float dw6_conv_1x1_bias_reo[96];
float dw6_conv_1x1_output_reo[96][20][40];


uint16 fix_conv_weight_1x1_all_16[10000][32][32];
uint16 fix_conv_weight_3x3_all_16[10000][32][3][3];
uint16 fix_bias_all_16[5000][32];

uint512 fix_conv_weight_1x1_all_512bit[1000][32];
uint512 fix_conv_weight_3x3_all_512bit[1000][3][3];
uint512 fix_bias_all_512bit[500];

uint512 DDR_dw1_pool_out_PL_burst[64/32*82*2*162*2];
uint512 DDR_dw2_pool_out_PL_burst[96/32*42*2*82*2];
uint512 DDR_buf_burst[128*44*84];


uint8  image_raw[3][160][320];	// 0~255 RGB raw data
uint8  image_raw_pad[3][162][322];	// 0~255 RGB raw data


uint8   image_raw_g[3][162*2][322*2];	// 0~255 RGB raw data for grouped 4 images
uint8   image_raw_g_burst[3*162*2*322*2];


float dw1_conv_3x3_output_stitch[3][160*2][320*2];
float dw1_conv_1x1_output_stitch[48][160*2][320*2];
float dw1_max_pool_output_stitch[48][80*2][160*2];
float dw2_conv_3x3_output_stitch[48][80*2][160*2];
float dw2_conv_1x1_output_stitch[96][80*2][160*2];
float dw2_max_pool_output_stitch[96][40*2][80*2];
float dw3_conv_3x3_output_stitch[96][40*2][80*2];
float dw3_conv_1x1_output_stitch[192][40*2][80*2];
float dw3_max_pool_output_stitch[192][20*2][40*2];
float dw4_conv_3x3_output_stitch[192][20*2][40*2];
float dw4_conv_1x1_output_stitch[384][20*2][40*2];
float dw5_conv_3x3_output_stitch[384][20*2][40*2];
float dw5_conv_1x1_output_stitch[512][20*2][40*2];

// cat dw3(ch:192 -> 768) and dw5(ch:512)
float dw6_conv_3x3_input_stitch[1280][20*2][40*2];
float dw6_conv_3x3_output_stitch[1280][20*2][40*2];
float dw6_conv_1x1_output_stitch[96][20*2][40*2];
float pw7_conv_1x1_output_stitch[10][20*2][40*2];

// reordered weights for the mysterious dw3(192->768)
float dw6_conv_3x3_input_reo_stitch[1280][20*2][40*2];
float dw6_conv_3x3_output_reo_stitch[1280][20*2][40*2];
float dw6_conv_1x1_output_reo_stitch[96][20*2][40*2];




void golden_model();
void reorder_weight_fix();


void stitch_outputs(int offset_h, int offset_w)
{
	for(int c = 0;c < 3; c++) {
		for(int h = 0; h < 160; h++) {
			for(int w = 0; w < 320; w++) {
				dw1_conv_3x3_output_stitch[c][h + 160*offset_h][w + 320*offset_w] = dw1_conv_3x3_output[c][h][w];
			}
		}
	}

	for(int c = 0;c < 48; c++) {
		for(int h = 0; h < 160; h++) {
			for(int w = 0; w < 320; w++) {
				dw1_conv_1x1_output_stitch[c][h + 160*offset_h][w + 320*offset_w] = dw1_conv_1x1_output[c][h][w];
			}
		}
	}


	for(int c = 0;c < 48; c++) {
		for(int h = 0; h < 80; h++) {
			for(int w = 0; w < 160; w++) {
				dw1_max_pool_output_stitch[c][h + 80*offset_h][w + 160*offset_w] = dw1_max_pool_output[c][h][w];
			}
		}
	}

	for(int c = 0;c < 48; c++) {
		for(int h = 0; h < 80; h++) {
			for(int w = 0; w < 160; w++) {
				dw2_conv_3x3_output_stitch[c][h + 80*offset_h][w + 160*offset_w] = dw2_conv_3x3_output[c][h][w];
			}
		}
	}

	for(int c = 0;c < 96; c++) {
		for(int h = 0; h < 80; h++) {
			for(int w = 0; w < 160; w++) {
				dw2_conv_1x1_output_stitch[c][h + 80*offset_h][w + 160*offset_w] = dw2_conv_1x1_output[c][h][w];
			}
		}
	}


	for(int c = 0;c < 96; c++) {
		for(int h = 0; h < 40; h++) {
			for(int w = 0; w < 80; w++) {
				dw2_max_pool_output_stitch[c][h + 40*offset_h][w + 80*offset_w] = dw2_max_pool_output[c][h][w];
			}
		}
	}


	for(int c = 0;c < 96; c++) {
		for(int h = 0; h < 40; h++) {
			for(int w = 0; w < 80; w++) {
				dw3_conv_3x3_output_stitch[c][h + 40*offset_h][w + 80*offset_w] = dw3_conv_3x3_output[c][h][w];
			}
		}
	}

	for(int c = 0;c < 192; c++) {
		for(int h = 0; h < 40; h++) {
			for(int w = 0; w < 80; w++) {
				dw3_conv_1x1_output_stitch[c][h + 40*offset_h][w + 80*offset_w] = dw3_conv_1x1_output[c][h][w];
			}
		}
	}


	for(int c = 0;c < 192; c++) {
		for(int h = 0; h < 20; h++) {
			for(int w = 0; w < 40; w++) {
				dw3_max_pool_output_stitch[c][h + 20*offset_h][w + 40*offset_w] = dw3_max_pool_output[c][h][w];
			}
		}
	}


	for(int c = 0;c < 192; c++) {
		for(int h = 0; h < 20; h++) {
			for(int w = 0; w < 40; w++) {
				dw4_conv_3x3_output_stitch[c][h + 20*offset_h][w + 40*offset_w] = dw4_conv_3x3_output[c][h][w];
			}
		}
	}

	for(int c = 0;c < 384; c++) {
		for(int h = 0; h < 20; h++) {
			for(int w = 0; w < 40; w++) {
				dw4_conv_1x1_output_stitch[c][h + 20*offset_h][w + 40*offset_w] = dw4_conv_1x1_output[c][h][w];
			}
		}
	}

	for(int c = 0;c < 384; c++) {
		for(int h = 0; h < 20; h++) {
			for(int w = 0; w < 40; w++) {
				dw5_conv_3x3_output_stitch[c][h + 20*offset_h][w + 40*offset_w] = dw5_conv_3x3_output[c][h][w];
			}
		}
	}

	for(int c = 0;c < 512; c++) {
		for(int h = 0; h < 20; h++) {
			for(int w = 0; w < 40; w++) {
				dw5_conv_1x1_output_stitch[c][h + 20*offset_h][w + 40*offset_w] = dw5_conv_1x1_output[c][h][w];
			}
		}
	}

	for(int c = 0;c < 1280; c++) {
		for(int h = 0; h < 20; h++) {
			for(int w = 0; w < 40; w++) {
				dw6_conv_3x3_output_stitch[c][h + 20*offset_h][w + 40*offset_w] = dw6_conv_3x3_output[c][h][w];
			}
		}
	}

	for(int c = 0;c < 96; c++) {
		for(int h = 0; h < 20; h++) {
			for(int w = 0; w < 40; w++) {
				dw6_conv_1x1_output_stitch[c][h + 20*offset_h][w + 40*offset_w] = dw6_conv_1x1_output[c][h][w];
			}
		}
	}


	for(int c = 0;c < 10; c++) {
		for(int h = 0; h < 20; h++) {
			for(int w = 0; w < 40; w++) {
				pw7_conv_1x1_output_stitch[c][h + 20*offset_h][w + 40*offset_w] = pw7_conv_1x1_output[c][h][w];
			}
		}
	}


	for(int c = 0;c < 1280; c++) {
		for(int h = 0; h < 20; h++) {
			for(int w = 0; w < 40; w++) {
				dw6_conv_3x3_output_reo_stitch[c][h + 20*offset_h][w + 40*offset_w] = dw6_conv_3x3_output_reo[c][h][w];
			}
		}
	}

	for(int c = 0;c < 96; c++) {
		for(int h = 0; h < 20; h++) {
			for(int w = 0; w < 40; w++) {
				dw6_conv_1x1_output_reo_stitch[c][h + 20*offset_h][w + 40*offset_w] = dw6_conv_1x1_output_reo[c][h][w];
			}
		}
	}
}

void load_weights()
{
    std::ifstream ifs_param("weights_floating.bin", std::ios::in | std::ios::binary);

    ///////////// Read Weights ///////////////////////
    ifs_param.read((char*)(**dw1_conv_3x3_weight), 3*3*3*sizeof(float));
    ifs_param.read((char*)dw1_conv_3x3_bias, 3*sizeof(float));
    ifs_param.read((char*)(*dw1_conv_1x1_weight), 48*3*sizeof(float));
    ifs_param.read((char*)dw1_conv_1x1_bias, 48*sizeof(float));
    ifs_param.read((char*)(**dw2_conv_3x3_weight), 48*3*3*sizeof(float));
    ifs_param.read((char*)dw2_conv_3x3_bias, 48*sizeof(float));
    ifs_param.read((char*)(*dw2_conv_1x1_weight), 96*48*sizeof(float));
    ifs_param.read((char*)dw2_conv_1x1_bias, 96*sizeof(float));
    ifs_param.read((char*)(**dw3_conv_3x3_weight), 96*3*3*sizeof(float));
    ifs_param.read((char*)dw3_conv_3x3_bias, 96*sizeof(float));
    ifs_param.read((char*)(*dw3_conv_1x1_weight), 192*96*sizeof(float));
    ifs_param.read((char*)dw3_conv_1x1_bias, 192*sizeof(float));
    ifs_param.read((char*)(**dw4_conv_3x3_weight), 192*3*3*sizeof(float));
    ifs_param.read((char*)dw4_conv_3x3_bias, 192*sizeof(float));
    ifs_param.read((char*)(*dw4_conv_1x1_weight), 384*192*sizeof(float));
    ifs_param.read((char*)dw4_conv_1x1_bias, 384*sizeof(float));
    ifs_param.read((char*)(*dw5_conv_3x3_weight), 384*3*3*sizeof(float));
    ifs_param.read((char*)dw5_conv_3x3_bias, 384*sizeof(float));
    ifs_param.read((char*)(*dw5_conv_1x1_weight), 512*384*sizeof(float));
    ifs_param.read((char*)dw5_conv_1x1_bias, 512*sizeof(float));
    ifs_param.read((char*)(*dw6_conv_3x3_weight), 1280*3*3*sizeof(float));
    ifs_param.read((char*)dw6_conv_3x3_bias, 1280*sizeof(float));
    ifs_param.read((char*)(*dw6_conv_1x1_weight), 96*1280*sizeof(float));
    ifs_param.read((char*)dw6_conv_1x1_bias, 96*sizeof(float));
    ifs_param.read((char*)(*pw7_conv_1x1_weight), 10*96*sizeof(float));
    ifs_param.close();
}


int test_one_group( char* img0, char* img1, char* img2, char* img3, char* img_g )
{
	///////////// GOLDEN C ///////////////////////////
	//////////////////////////////////////////////////


    ///////////// Prepare Image //////////////////////
	std::ifstream ifs_image_raw0(img0, std::ios::in | std::ios::binary);
    ifs_image_raw0.read((char*)(**image_raw), 3*160*320*sizeof(uint8));
    ///////////////// IMAGE NORM ///////////////////
	for(int j = 0; j < 160; j++) {
		for(int k = 0; k < 320; k++) {
			image[0][j][k] = (((image_raw[0][j][k].to_int()/255.0)-0.5)/0.25);
			image[1][j][k] = (((image_raw[1][j][k].to_int()/255.0)-0.5)/0.25);
			image[2][j][k] = (((image_raw[2][j][k].to_int()/255.0)-0.5)/0.25);
		}
	}
    /////// GOLDEN MODEL ///////////
    golden_model();
    stitch_outputs(0, 0);

    ///////////// Prepare Image //////////////////////
	std::ifstream ifs_image_raw1(img1, std::ios::in | std::ios::binary);
    ifs_image_raw1.read((char*)(**image_raw), 3*160*320*sizeof(uint8));
    ///////////////// IMAGE NORM ///////////////////
	for(int j = 0; j < 160; j++) {
		for(int k = 0; k < 320; k++) {
			image[0][j][k] = (((image_raw[0][j][k].to_int()/255.0)-0.5)/0.25);
			image[1][j][k] = (((image_raw[1][j][k].to_int()/255.0)-0.5)/0.25);
			image[2][j][k] = (((image_raw[2][j][k].to_int()/255.0)-0.5)/0.25);
		}
	}
    /////// GOLDEN MODEL ///////////
    golden_model();
    stitch_outputs(0, 1);

    ///////////// Prepare Image //////////////////////
	std::ifstream ifs_image_raw2(img2, std::ios::in | std::ios::binary);
    ifs_image_raw2.read((char*)(**image_raw), 3*160*320*sizeof(uint8));
    ///////////////// IMAGE NORM ///////////////////
	for(int j = 0; j < 160; j++) {
		for(int k = 0; k < 320; k++) {
			image[0][j][k] = (((image_raw[0][j][k].to_int()/255.0)-0.5)/0.25);
			image[1][j][k] = (((image_raw[1][j][k].to_int()/255.0)-0.5)/0.25);
			image[2][j][k] = (((image_raw[2][j][k].to_int()/255.0)-0.5)/0.25);
		}
	}
    /////// GOLDEN MODEL ///////////
    golden_model();
    stitch_outputs(1, 0);

    ///////////// Prepare Image //////////////////////
	std::ifstream ifs_image_raw3(img3, std::ios::in | std::ios::binary);
    ifs_image_raw3.read((char*)(**image_raw), 3*160*320*sizeof(uint8));
    ///////////////// IMAGE NORM ///////////////////
	for(int j = 0; j < 160; j++) {
		for(int k = 0; k < 320; k++) {
			image[0][j][k] = (((image_raw[0][j][k].to_int()/255.0)-0.5)/0.25);
			image[1][j][k] = (((image_raw[1][j][k].to_int()/255.0)-0.5)/0.25);
			image[2][j][k] = (((image_raw[2][j][k].to_int()/255.0)-0.5)/0.25);
		}
	}
    /////// GOLDEN MODEL ///////////
    golden_model();
    stitch_outputs(1, 1);


    ///////////////////// HLS ///////////////////////////
    /////////////////////////////////////////////////////
    reorder_weight_fix();

	std::ifstream ifs_image_raw_g(img_g, std::ios::in | std::ios::binary);
    ifs_image_raw_g.read((char*)(image_raw_g_burst), 3*162*2*322*2*sizeof(uint8));

    float predict_box[4][5];
    int constant[4][3];	  //conf_j, conf_m, conf_n
    float debug[10][32][44][84];
    float box[4] = {1.4940052559648322, 2.3598481287086823, 4.0113013115312155, 5.760873975661669};

    SkyNet(image_raw_g_burst,

    		fix_conv_weight_1x1_all_512bit,
    		fix_conv_weight_3x3_all_512bit,
			fix_bias_all_512bit,

			DDR_dw1_pool_out_PL_burst,
			DDR_dw2_pool_out_PL_burst,
			DDR_buf_burst,

			debug,
			predict_box,
			constant
			);



    int h = 20;
    int w = 40;

    printf("PL Outputs:\n");

    for(int img = 0; img < 4; img++) {
    	printf("Image ID %d\n", img);

    	predict_box[img][0] = 1 / (1 + exp(-predict_box[img][0])) + constant[img][1];
    	predict_box[img][1] = 1 / (1 + exp(-predict_box[img][1])) + constant[img][2];
    	if( constant[img][0] == 0 ) {
    		predict_box[img][2] = exp(predict_box[img][2]) * box[0];
    		predict_box[img][3] = exp(predict_box[img][3]) * box[1];
    	}
    	else {
    		predict_box[img][2] = exp(predict_box[img][2]) * box[2];
    		predict_box[img][3] = exp(predict_box[img][3]) * box[3];
    	}
    	predict_box[img][4] = 1 / (1 + exp(-predict_box[img][4]));


		printf("%f %f %f %f %f\n", predict_box[img][0] / w, predict_box[img][1] / h,
				predict_box[img][2] / w, predict_box[img][3] / h, predict_box[img][4]);

		int x1, y1, x2, y2;
		predict_box[img][0] = predict_box[img][0] / w;
		predict_box[img][1] = predict_box[img][1] / h;
		predict_box[img][2] = predict_box[img][2] / w;
		predict_box[img][3] = predict_box[img][3] / h;

		x1 = (unsigned int)(((predict_box[img][0] - predict_box[img][2]/2.0) * 640));
		y1 = (unsigned int)(((predict_box[img][1] - predict_box[img][3]/2.0) * 360));
		x2 = (unsigned int)(((predict_box[img][0] + predict_box[img][2]/2.0) * 640));
		y2 = (unsigned int)(((predict_box[img][1] + predict_box[img][3]/2.0) * 360));

		printf("boxes: %d %d %d %d\n", x1, y1, x2, y2);
    }

    return 0;
}



int main()
{

	load_weights();

	printf("Testing on image 0.jpg to 3.jpg\n");
	test_one_group("0.bin", "1.bin", "2.bin", "3.bin", "stitched_0_3.bin");
//
//	printf("Testing on image 4.jpg to 7.jpg\n");
//	test_one_group("4.bin", "5.bin", "6.bin", "7.bin", "stitched_4_7.bin");
//
//	printf("Testing on image 8.jpg to 11.jpg\n");
//	test_one_group("8.bin", "9.bin", "10.bin", "11.bin", "stitched_8_11.bin");


	return 0;

}





