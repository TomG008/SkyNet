
#include "hls_stream.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <ap_fixed.h>
#include <stdio.h>
#include "net_hls.h"


using namespace std;


/* floating point input weights */
extern float dw1_conv_3x3_weight[3][3][3];
extern float dw1_conv_3x3_bias[3];

extern float dw1_conv_1x1_weight[48][3];
extern float dw1_conv_1x1_bias[48];

extern float dw2_conv_3x3_weight[48][3][3];
extern float dw2_conv_3x3_bias[48];

extern float dw2_conv_1x1_weight[96][48];
extern float dw2_conv_1x1_bias[96];

extern float dw3_conv_3x3_weight[96][3][3];
extern float dw3_conv_3x3_bias[96];

extern float dw3_conv_1x1_weight[192][96];
extern float dw3_conv_1x1_bias[192];

extern float dw4_conv_3x3_weight[192][3][3];
extern float dw4_conv_3x3_bias[192];

extern float dw4_conv_1x1_weight[384][192];
extern float dw4_conv_1x1_bias[384];

extern float dw5_conv_3x3_weight[384][3][3];
extern float dw5_conv_3x3_bias[384];

extern float dw5_conv_1x1_weight[512][384];
extern float dw5_conv_1x1_bias[512];

extern float dw6_conv_3x3_weight[1280][3][3];
extern float dw6_conv_3x3_bias[1280];

extern float dw6_conv_1x1_weight[96][1280];
extern float dw6_conv_1x1_bias[96];

extern float pw7_conv_1x1_weight[10][96];

// reordered weights for the mysterious dw3(192->768)
extern float dw6_conv_3x3_weight_reo[1280][3][3];
extern float dw6_conv_3x3_bias_reo[1280];
extern float dw6_conv_3x3_input_reo[1280][20][40];
extern float dw6_conv_3x3_output_reo[1280][20][40];

extern float dw6_conv_1x1_weight_reo[96][1280];
extern float dw6_conv_1x1_bias_reo[96];
extern float dw6_conv_1x1_output_reo[96][20][40];


/* fixed point parameters */
FIX_WT fix_dw1_conv_3x3_weight[32][3][3];	// 3 -> 32
FIX_WT fix_dw1_conv_3x3_bias[32];			// 3 -> 32

FIX_WT fix_dw1_conv_1x1_weight[64][32];		// 3 -> 32, 48 -> 64
FIX_WT fix_dw1_conv_1x1_bias[64];			// 48 -> 64

FIX_WT fix_dw2_conv_3x3_weight[64][3][3];   // 48 -> 64
FIX_WT fix_dw2_conv_3x3_bias[64];           // 48 -> 64

FIX_WT fix_dw2_conv_1x1_weight[96][64];     // 48 -> 64
FIX_WT fix_dw2_conv_1x1_bias[96];

FIX_WT fix_dw3_conv_3x3_weight[96][3][3];
FIX_WT fix_dw3_conv_3x3_bias[96];

FIX_WT fix_dw3_conv_1x1_weight[192][96];
FIX_WT fix_dw3_conv_1x1_bias[192];

FIX_WT fix_dw4_conv_3x3_weight[192][3][3];
FIX_WT fix_dw4_conv_3x3_bias[192];

FIX_WT fix_dw4_conv_1x1_weight[384][192];
FIX_WT fix_dw4_conv_1x1_bias[384];

FIX_WT fix_dw5_conv_3x3_weight[384][3][3];
FIX_WT fix_dw5_conv_3x3_bias[384];

FIX_WT fix_dw5_conv_1x1_weight[512][384];
FIX_WT fix_dw5_conv_1x1_bias[512];

FIX_WT fix_dw6_conv_3x3_weight[1280][3][3];
FIX_WT fix_dw6_conv_3x3_bias[1280];

FIX_WT fix_dw6_conv_1x1_weight[96][1280];
FIX_WT fix_dw6_conv_1x1_bias[96];

FIX_WT fix_pw7_conv_1x1_weight[32][96];		// 10 -> 32



extern uint16 fix_conv_weight_1x1_all_16[10000][32][32];
extern uint16 fix_conv_weight_3x3_all_16[10000][32][3][3];
extern uint16 fix_bias_all_16[5000][32];

FIX_WT fix_conv_weight_1x1_all[10000][32][32];
FIX_WT fix_conv_weight_3x3_all[10000][32][3][3];
FIX_WT fix_bias_all[5000][32];

extern uint512 fix_conv_weight_1x1_all_512bit[1000][32];
extern uint512 fix_conv_weight_3x3_all_512bit[1000][3][3];
extern uint512 fix_bias_all_512bit[500];


void reorder_weight_fix()
{

    // reorder dw6_conv_3x3_weight and dw6_conv_3x3_bias
    for(int ch = 0; ch < 768; ch += 4) {
        dw6_conv_3x3_bias_reo[ch] = dw6_conv_3x3_bias[ ch / 4];
        dw6_conv_3x3_bias_reo[ch + 1] = dw6_conv_3x3_bias[ ch / 4 + 192];
        dw6_conv_3x3_bias_reo[ch + 2] = dw6_conv_3x3_bias[ ch / 4 + 192 * 2];
        dw6_conv_3x3_bias_reo[ch + 3] = dw6_conv_3x3_bias[ ch / 4 + 192 * 3];

        for(int m = 0; m < 3; m++) {
            for(int n = 0; n < 3; n++) {
                dw6_conv_3x3_weight_reo[ch    ][m][n] = dw6_conv_3x3_weight[ ch / 4 ][m][n];
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


    //for dw1_conv_3x3
    for(int m = 0; m < 3; m++) {
    	for(int n = 0; n < 3; n++) {
    		for(int c = 0; c < 32; c++) {
    			if(c < 3) {
    				fix_dw1_conv_3x3_weight[c][m][n] = (FIX_WT)dw1_conv_3x3_weight[c][m][n];
    				fix_dw1_conv_3x3_bias[c] = (FIX_WT)dw1_conv_3x3_bias[c];
    			}
    			else {
    				fix_dw1_conv_3x3_weight[c][m][n] = 0;
    				fix_dw1_conv_3x3_bias[c] = 0;
    			}
    		}
    	}
    }

    //for dw1_conv_1x1
    for(int co = 0; co < 64; co++) {
    	if(co < 48) {
    		fix_dw1_conv_1x1_bias[co] = dw1_conv_1x1_bias[co];
    	}
    	else {
    		fix_dw1_conv_1x1_bias[co] = 0;
    	}

    	for(int ci = 0; ci < 32; ci++) {
    		if(ci < 3) {
    			fix_dw1_conv_1x1_weight[co][ci] = (FIX_WT)dw1_conv_1x1_weight[co][ci];
    		}
    		else {
    			fix_dw1_conv_1x1_weight[co][ci] = 0;
    		}
    	}
    }

    //for dw2_conv_3x3
    for(int c = 0; c < 64; c++) {
    	if(c < 48) {
    		fix_dw2_conv_3x3_bias[c] = (FIX_WT)dw2_conv_3x3_bias[c];
    	}
    	else {
    		fix_dw2_conv_3x3_bias[c] = 0;
    	}

		for(int m = 0; m < 3; m++) {
			for(int n = 0; n < 3; n++) {
				fix_dw2_conv_3x3_weight[c][m][n] = (FIX_WT)dw2_conv_3x3_weight[c][m][n];
			}
		}
	}

    //for dw2_conv_1x1
    for(int co = 0; co < 96; co++) {
		fix_dw2_conv_1x1_bias[co] = (FIX_WT)dw2_conv_1x1_bias[co];
		for(int ci = 0; ci < 64; ci++) {
			if(ci < 48) {
				fix_dw2_conv_1x1_weight[co][ci] = (FIX_WT)dw2_conv_1x1_weight[co][ci];
			}
			else {
				fix_dw2_conv_1x1_weight[co][ci] = 0;
			}
		}
	}

    //for dw3_conv_3x3
	for(int c = 0; c < 96; c++) {
		fix_dw3_conv_3x3_bias[c] = (FIX_WT)dw3_conv_3x3_bias[c];
		for(int m = 0; m < 3; m++) {
			for(int n = 0; n < 3; n++) {
				fix_dw3_conv_3x3_weight[c][m][n] = (FIX_WT)dw3_conv_3x3_weight[c][m][n];
			}
		}
	}

    //for dw3_conv_1x1
	for(int co = 0; co < 192; co++) {
		fix_dw3_conv_1x1_bias[co] = (FIX_WT)dw3_conv_1x1_bias[co];
		for(int ci = 0; ci < 96; ci++) {
			fix_dw3_conv_1x1_weight[co][ci] = (FIX_WT)dw3_conv_1x1_weight[co][ci];
		}
	}

	//for dw4_conv_3x3
	for(int c = 0; c < 192; c++) {
		fix_dw4_conv_3x3_bias[c] = (FIX_WT)dw4_conv_3x3_bias[c];
		for(int m = 0; m < 3; m++) {
			for(int n = 0; n < 3; n++) {
				fix_dw4_conv_3x3_weight[c][m][n] = (FIX_WT)dw4_conv_3x3_weight[c][m][n];
			}
		}
	}

	//for dw4_conv_1x1
	for(int co = 0; co < 384; co++) {
		fix_dw4_conv_1x1_bias[co] = (FIX_WT)dw4_conv_1x1_bias[co];
		for(int ci = 0; ci < 192; ci++) {
			fix_dw4_conv_1x1_weight[co][ci] = (FIX_WT)dw4_conv_1x1_weight[co][ci];
		}
	}

	//for dw5_conv_3x3
	for(int c = 0; c < 384; c++) {
		fix_dw5_conv_3x3_bias[c] = (FIX_WT)dw5_conv_3x3_bias[c];
		for(int m = 0; m < 3; m++) {
			for(int n = 0; n < 3; n++) {
				fix_dw5_conv_3x3_weight[c][m][n] = (FIX_WT)dw5_conv_3x3_weight[c][m][n];
			}
		}
	}

	//for dw5_conv_1x1
	for(int co = 0; co < 512; co++) {
		for(int ci = 0; ci < 384; ci++) {
			if(co < 512) {
				fix_dw5_conv_1x1_weight[co][ci] = (FIX_WT)dw5_conv_1x1_weight[co][ci];
				fix_dw5_conv_1x1_bias[co] = (FIX_WT)dw5_conv_1x1_bias[co];
			}
			else {
				fix_dw5_conv_1x1_weight[co][ci] = 0;
				fix_dw5_conv_1x1_bias[co] = 0;
			}
		}
	}

	//for dw6_conv_3x3
	for(int c = 0; c < 1280; c++) {
		fix_dw6_conv_3x3_bias[c] = (FIX_WT)dw6_conv_3x3_bias_reo[c];
		for(int m = 0; m < 3; m++) {
			for(int n = 0; n < 3; n++) {
				fix_dw6_conv_3x3_weight[c][m][n] = (FIX_WT)dw6_conv_3x3_weight_reo[c][m][n];
			}
		}
	}

	//for dw6_conv_1x1
	for(int co = 0; co < 96; co++) {
		fix_dw6_conv_1x1_bias[co] = (FIX_WT)dw6_conv_1x1_bias_reo[co];
		for(int ci = 0; ci < 1280; ci++) {
			if(ci < 1280) {
				fix_dw6_conv_1x1_weight[co][ci] = (FIX_WT)dw6_conv_1x1_weight_reo[co][ci];
			}
			else {
				fix_dw6_conv_1x1_weight[co][ci] = 0;
			}
		}
	}

	//for pw7_conv_1x1
	for(int co = 0; co < 32; co++) {
		for(int ci = 0; ci < 96; ci++) {
			if(co < 10) {
				fix_pw7_conv_1x1_weight[co][ci] = (FIX_WT)pw7_conv_1x1_weight[co][ci];
			}
			else {
				fix_pw7_conv_1x1_weight[co][ci] = 0;
			}
		}
	}

	//////////// reorder conv_1x1 weights, and put all weights together

	// copy conv_1 to conv_weight_3x3_all
	int index_3x3 = -1;
	int index_1x1 = -1;
	int index_bias = -1;
	int CO_N, CI_N;

	// dw1_conv_3x3 weights and bias
	for(int c = 0; c < 32; c++) {
		if( c % 32 == 0) {
			index_3x3++;
			index_bias++;
		}
		fix_bias_all[index_bias][c] = fix_dw1_conv_3x3_bias[c];
		for(int m = 0; m < 3; m++) {
			for(int n = 0; n < 3; n++)
				fix_conv_weight_3x3_all[index_3x3][c][m][n] = fix_dw1_conv_3x3_weight[c][m][n];
		}
	}

	// dw1_conv_1x1 weights (reorder) and bias
	CO_N = 64 / 32;
	CI_N = 32 / 32;
	for(int CO = 0; CO < CO_N; CO++) {
		for(int CI = 0; CI < CI_N; CI++) {
			index_1x1++;

			for(int co = 0; co < 32; co++) {
				for(int ci = 0; ci < 32; ci++) {
					fix_conv_weight_1x1_all[index_1x1][co][ci] = fix_dw1_conv_1x1_weight[co + CO * 32][ci + CI * 32];
				}
			}
		}
	}
	for(int CO = 0; CO < CO_N; CO++) {
		index_bias++;
		for(int co = 0; co < 32; co++) {
			fix_bias_all[index_bias][co] = fix_dw1_conv_1x1_bias[co + CO * 32];
		}
	}

	// dw2_conv_3x3 weights and bias
	for(int c = 0; c < 64; c++) {
		if( c % 32 == 0) {
			index_3x3++;
			index_bias++;
		}
		fix_bias_all[index_bias][c % 32] = fix_dw2_conv_3x3_bias[c];
		for(int m = 0; m < 3; m++) {
			for(int n = 0; n < 3; n++)
				fix_conv_weight_3x3_all[index_3x3][c % 32][m][n] = fix_dw2_conv_3x3_weight[c][m][n];
		}
	}


	// dw2_conv_1x1 weights (reorder) and bias
	CO_N = 96 / 32;
	CI_N = 64 / 32;
	for(int CO = 0; CO < CO_N; CO++) {
		for(int CI = 0; CI < CI_N; CI++) {
			index_1x1++;

			for(int co = 0; co < 32; co++) {
				for(int ci = 0; ci < 32; ci++) {
					fix_conv_weight_1x1_all[index_1x1][co][ci] = fix_dw2_conv_1x1_weight[co + CO * 32][ci + CI * 32];
				}
			}
		}
	}
	for(int CO = 0; CO < CO_N; CO++) {
		index_bias++;
		for(int co = 0; co < 32; co++) {
			fix_bias_all[index_bias][co] = dw2_conv_1x1_bias[co + CO * 32];
		}
	}


	// dw3_conv_3x3 weights and bias
	for(int c = 0; c < 96; c++) {
		if( c % 32 == 0) {
			index_3x3++;
			index_bias++;
		}
		fix_bias_all[index_bias][c % 32] = fix_dw3_conv_3x3_bias[c];
		for(int m = 0; m < 3; m++) {
			for(int n = 0; n < 3; n++)
				fix_conv_weight_3x3_all[index_3x3][c % 32][m][n] = fix_dw3_conv_3x3_weight[c][m][n];
		}
	}


	// dw3_conv_1x1 weights (reorder) and bias
	CO_N = 192 / 32;
	CI_N = 96 / 32;
	for(int CO = 0; CO < CO_N; CO++) {
		for(int CI = 0; CI < CI_N; CI++) {
			index_1x1++;

			for(int co = 0; co < 32; co++) {
				for(int ci = 0; ci < 32; ci++) {
					fix_conv_weight_1x1_all[index_1x1][co][ci] = fix_dw3_conv_1x1_weight[co + CO * 32][ci + CI * 32];
				}
			}
		}
	}
	for(int CO = 0; CO < CO_N; CO++) {
		index_bias++;
		for(int co = 0; co < 32; co++) {
			fix_bias_all[index_bias][co] = dw3_conv_1x1_bias[co + CO * 32];
		}
	}


	// dw4_conv_3x3 weights and bias
	for(int c = 0; c < 192; c++) {
		if( c % 32 == 0) {
			index_3x3++;
			index_bias++;
		}
		fix_bias_all[index_bias][c % 32] = fix_dw4_conv_3x3_bias[c];
		for(int m = 0; m < 3; m++) {
			for(int n = 0; n < 3; n++)
				fix_conv_weight_3x3_all[index_3x3][c % 32][m][n] = fix_dw4_conv_3x3_weight[c][m][n];
		}
	}


	// dw4_conv_1x1 weights (reorder) and bias
	CO_N = 384 / 32;
	CI_N = 192 / 32;
	for(int CO = 0; CO < CO_N; CO++) {
		for(int CI = 0; CI < CI_N; CI++) {
			index_1x1++;

			for(int co = 0; co < 32; co++) {
				for(int ci = 0; ci < 32; ci++) {
					fix_conv_weight_1x1_all[index_1x1][co][ci] = fix_dw4_conv_1x1_weight[co + CO * 32][ci + CI * 32];
				}
			}
		}
	}
	for(int CO = 0; CO < CO_N; CO++) {
		index_bias++;
		for(int co = 0; co < 32; co++) {
			fix_bias_all[index_bias][co] = dw4_conv_1x1_bias[co + CO * 32];
		}
	}


	// dw5_conv_3x3 weights and bias
	for(int c = 0; c < 384; c++) {
		if( c % 32 == 0) {
			index_3x3++;
			index_bias++;
		}
		fix_bias_all[index_bias][c % 32] = fix_dw5_conv_3x3_bias[c];
		for(int m = 0; m < 3; m++) {
			for(int n = 0; n < 3; n++)
				fix_conv_weight_3x3_all[index_3x3][c % 32][m][n] = fix_dw5_conv_3x3_weight[c][m][n];
		}
	}


	// dw5_conv_1x1 weights (reorder) and bias
	CO_N = 512 / 32;
	CI_N = 384 / 32;
	for(int CO = 0; CO < CO_N; CO++) {
		for(int CI = 0; CI < CI_N; CI++) {
			index_1x1++;

			for(int co = 0; co < 32; co++) {
				for(int ci = 0; ci < 32; ci++) {
					fix_conv_weight_1x1_all[index_1x1][co][ci] = fix_dw5_conv_1x1_weight[co + CO * 32][ci + CI * 32];
				}
			}
		}
	}
	for(int CO = 0; CO < CO_N; CO++) {
		index_bias++;
		for(int co = 0; co < 32; co++) {
			fix_bias_all[index_bias][co] = dw5_conv_1x1_bias[co + CO * 32];
		}
	}


	// dw6_conv_3x3 weights and bias
	for(int c = 0; c < 1280; c++) {
		if( c % 32 == 0) {
			index_3x3++;
			index_bias++;
		}
		fix_bias_all[index_bias][c % 32] = fix_dw6_conv_3x3_bias[c];
		for(int m = 0; m < 3; m++) {
			for(int n = 0; n < 3; n++)
				fix_conv_weight_3x3_all[index_3x3][c % 32][m][n] = fix_dw6_conv_3x3_weight[c][m][n];
		}
	}


	// dw6_conv_1x1 weights (reorder) and bias
	CO_N = 96 / 32;
	CI_N = 1280 / 32;
	for(int CO = 0; CO < CO_N; CO++) {
		for(int CI = 0; CI < CI_N; CI++) {
			index_1x1++;

			for(int co = 0; co < 32; co++) {
				for(int ci = 0; ci < 32; ci++) {
					fix_conv_weight_1x1_all[index_1x1][co][ci] = fix_dw6_conv_1x1_weight[co + CO * 32][ci + CI * 32];
				}
			}
		}
	}
	for(int CO = 0; CO < CO_N; CO++) {
		index_bias++;
		for(int co = 0; co < 32; co++) {
			fix_bias_all[index_bias][co] = dw6_conv_1x1_bias[co + CO * 32];
		}
	}


	// pw7_conv_1x1 weights (reorder)
	CO_N = 32 / 32;
	CI_N = 96 / 32;
	for(int CO = 0; CO < CO_N; CO++) {
		for(int CI = 0; CI < CI_N; CI++) {
			index_1x1++;

			for(int co = 0; co < 32; co++) {
				for(int ci = 0; ci < 32; ci++) {
					fix_conv_weight_1x1_all[index_1x1][co][ci] = fix_pw7_conv_1x1_weight[co + CO * 32][ci + CI * 32];
				}
			}
		}
	}


	index_3x3++;
	index_1x1++;
	index_bias++;

//	printf("# of 3x3 weight chunks %d\n", index_3x3);
//	printf("# of 1x1 weight chunks: %d\n", index_1x1);
//	printf("# of bias chunks: %d\n", index_bias);


    for(int i = 0; i < 10000; i++) {
    	for(int m = 0; m < 32; m++) {
    		for(int n = 0; n < 32; n++) {
    			uint16 DATA = 0;
    			DATA.range(10, 0) = fix_conv_weight_1x1_all[i][m][n].range(10, 0);
    			fix_conv_weight_1x1_all_16[i][m][n].range(15, 0) = DATA.range(15, 0);
    		}
    	}
    }


    for(int i = 0; i < 10000; i++) {
    	for(int k = 0; k < 32; k++) {
			for(int m = 0; m < 3; m++) {
				for(int n = 0; n < 3; n++) {
					uint16 DATA = 0;
					DATA.range(10, 0) = fix_conv_weight_3x3_all[i][k][m][n].range(10, 0);
					fix_conv_weight_3x3_all_16[i][k][m][n].range(15, 0) = DATA.range(15, 0);
				}
			}
    	}
    }

    for(int i = 0; i < 5000; i++) {
    	for(int k = 0; k < 32; k++) {
    		uint16 DATA = 0;
    		DATA.range(10, 0) = fix_bias_all[i][k].range(10, 0);
    		fix_bias_all_16[i][k].range(15, 0) = DATA.range(15, 0);
    	}
    }



//    /// Write weights into 16-bit bin file
//    /// One clock cycle one weight (16-bit)
//    std::ofstream ofs_param_write_16("weights_fixed_16bit.bin", std::ios::out | std::ios::binary);
//	// write conv_1x1 weights into weight bin
//	ofs_param_write_16.write((char*)fix_conv_weight_1x1_all_16, index_1x1 * 32 * 32 * sizeof(uint16));
//	// write conv_3x3 weights into weight bin
//	ofs_param_write_16.write((char*)fix_conv_weight_3x3_all_16, index_3x3 * 32 * 3 * 3 * sizeof(uint16));
//	// write bias_all into weight bin
//	ofs_param_write_16.write((char*)fix_bias_all_16, index_bias * 32 *sizeof(uint16));
//	ofs_param_write_16.close();



	/// Write weights into 512-bit bin file
	/// One clock cycle 32 weights (16-bit * 32)
	std::ofstream ofs_param_write_512("weights_fixed.bin", std::ios::out | std::ios::binary);

    // fill fix_conv_weight_1x1_all into 512 bit-width bus
    for(int i = 0; i < index_1x1; i++) {
    	for(int k = 0; k < 32; k++) {

    		uint512 DATA = 0;
    		for(int j = 0; j < 32; j ++) {
    			DATA.range(j*16 + WT_RG, j*16)      = fix_conv_weight_1x1_all[i][j][k].range(WT_RG, 0);
    		}
    		fix_conv_weight_1x1_all_512bit[i][k].range(511, 0) = DATA.range(511, 0);
    	}
    }
    ofs_param_write_512.write((char*)fix_conv_weight_1x1_all_512bit, index_1x1 * 32 * sizeof(uint512));

    // fill fix_conv_weight_3x3_all into 512 bit-width bus
    for(int i = 0; i < index_3x3; i++) {
    	for(int m = 0; m < 3; m++) {
    		for(int n = 0; n < 3; n++) {

    			uint512 DATA = 0;
				for(int j = 0; j < 32; j++) {
					DATA.range(j*16 + WT_RG, j*16) = fix_conv_weight_3x3_all[i][j][m][n].range(WT_RG, 0);
				}
				fix_conv_weight_3x3_all_512bit[i][m][n].range(511, 0) = DATA.range(511, 0);
    		}
    	}
    }
    ofs_param_write_512.write((char*)fix_conv_weight_3x3_all_512bit, index_3x3 * 3 * 3 * sizeof(uint512));

    // fill fix_bias_all into 512 bit-width bus
    for(int i = 0; i < index_bias; i++) {
    	uint512 DATA = 0;
		for(int j = 0; j < 32; j++) {
			DATA.range(j*16 + WT_RG, j*16) = fix_bias_all[i][j].range(WT_RG, 0);
		}
		fix_bias_all_512bit[i].range(511, 0) = DATA.range(511, 0);
    }
    ofs_param_write_512.write((char*)fix_bias_all_512bit, index_bias * sizeof(uint512));
    ofs_param_write_512.close();

}

