#include "composed_pedal.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint32_t u32le(const unsigned char *p){return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);}
static void put16(FILE*f,uint16_t v){fputc(v&255,f);fputc(v>>8,f);}
static void put32(FILE*f,uint32_t v){put16(f,(uint16_t)v);put16(f,(uint16_t)(v>>16));}

int main(int argc,char**argv){
 if(argc<3||argc>4){fprintf(stderr,"input.wav output.wav [input_peak_v]\n");return 2;}float input_peak=argc==4?strtof(argv[3],0):0.1F;
 FILE*f=fopen(argv[1],"rb");if(!f){perror(argv[1]);return 2;}
 unsigned char h[44];if(fread(h,1,44,f)!=44||u32le(h+24)!=48000||h[22]!=1||h[34]!=16){fprintf(stderr,"expected mono PCM16 48k WAV\n");return 2;}
 uint32_t n=u32le(h+40)/2;int16_t*pcm=malloc(n*sizeof(*pcm));float*out=malloc(n*sizeof(*out));if(!pcm||!out)return 2;
 if(fread(pcm,sizeof(*pcm),n,f)!=n)return 2;fclose(f);
 double mean=0;for(uint32_t i=0;i<n;i++)mean+=pcm[i];mean/=n;float peak=0;
 for(uint32_t i=0;i<n;i++){float x=fabsf((float)pcm[i]-(float)mean);if(x>peak)peak=x;}
 composed_pedal_init();float maxabs=0;double tail=0;
 uint32_t previous_holds=0;for(uint32_t i=0;i<n;i++){float x=((float)pcm[i]-(float)mean)*(input_peak/peak);out[i]=composed_pedal_process_sample(x);uint32_t holds=composed_pedal_hold_count(),fault=composed_pedal_fault_stage();if(holds!=previous_holds&&previous_holds==0)fprintf(stderr,"first hold at %u port %.9g\n",i,composed_pedal_last_port());previous_holds=holds;if(fault!=0U){fprintf(stderr,"fault at %u stage %u port %.9g\n",i,fault,composed_pedal_last_port());return 3;}if(!isfinite(out[i])){fprintf(stderr,"nonfinite at %u stage %u value %.9g\n",i,fault,out[i]);return 3;}}
 uint32_t tailn=n<4800?n:4800;for(uint32_t i=n-tailn;i<n;i++)tail+=out[i];float dc=(float)(tail/tailn);
 for(uint32_t i=0;i<n;i++){out[i]-=dc;float a=fabsf(out[i]);if(a>maxabs)maxabs=a;}
 uint32_t fade=n<480?n:480;for(uint32_t i=0;i<fade;i++)out[n-fade+i]*=(float)(fade-1-i)/(float)fade;
 f=fopen(argv[2],"wb");if(!f){perror(argv[2]);return 2;}uint32_t bytes=n*2;fwrite("RIFF",1,4,f);put32(f,36+bytes);fwrite("WAVEfmt ",1,8,f);put32(f,16);put16(f,1);put16(f,1);put32(f,48000);put32(f,96000);put16(f,2);put16(f,16);fwrite("data",1,4,f);put32(f,bytes);
 for(uint32_t i=0;i<n;i++){float y=out[i]*(0.95F/maxabs);if(y>1)y=1;if(y< -1)y=-1;put16(f,(uint16_t)(int16_t)lrintf(y*32767));}fclose(f);
 printf("samples=%u peak_v=%.9g dc_v=%.9g fallbacks=%u adaptive=%u holds=%u q2_linear=[%.9g,%.9g]\n",n,maxabs,dc,composed_pedal_fallback_count(),composed_pedal_adaptive_count(),composed_pedal_hold_count(),composed_pedal_q2_linear_min(),composed_pedal_q2_linear_max());free(out);free(pcm);return 0;
}
