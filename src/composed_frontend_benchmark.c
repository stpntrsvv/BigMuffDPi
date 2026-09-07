#if defined(COMPOSED_FRONTEND_BENCHMARK) || defined(COMPOSED_PEDAL_RUNTIME) || defined(COMPOSED_PEDAL_HOST)
#include <stdint.h>
#ifdef COMPOSED_PEDAL_HOST
#include <math.h>
#else
#include "stm32g4xx.h"
#endif
#ifdef COMPOSED_GENERALIZED_ALPHA02
#include "composed_frontend_fixture_alpha02.h"
#elif defined(COMPOSED_BDF2)
#include "composed_frontend_fixture_bdf2.h"
#else
#include "composed_frontend_fixture.h"
#endif
#include "composed_pedal.h"
#ifdef COMPOSED_FRONTEND_BENCHMARK
#include "port_multirate_fixture.h"
#endif

#define PASSES 8U
#define FS 25.8649e-3F
#define DS (1.9F*25.8649e-3F)
#define IS 10.025e-15F
#define DIS2 4.0e-9F
#ifndef COMPOSED_PORT_RELAX
#define COMPOSED_PORT_RELAX .5F
#endif
#ifndef COMPOSED_Q1_DIRECT_EXACT_THRESHOLD
#define COMPOSED_Q1_DIRECT_EXACT_THRESHOLD 1.0F
#endif
typedef union { float f; uint32_t u; } bits_t;
typedef struct { float as[7], as_old[7], aq[3], bs[2], bs_old[2], bq, base, base_old, output;
 float av[3],ac[3],bv,bc,b_linear,b_linear_min,b_linear_max,ss[4],sq[2],sv[2],sc[2]
#ifdef COMPOSED_Q1_DIRECT_EXP
 ,se[2]
#endif
 ,last_port,port_offset,previous_input;
#if defined(COMPOSED_BDF2) || defined(COMPOSED_GENERALIZED_ALPHA02)
 float ss_old[4];
#endif
 uint32_t index,fallbacks,adaptive,slow_adaptive,q1_exact[2],q2_bracketed,holds,fault; } state_t;
#ifdef COMPOSED_PEDAL_HOST_DIAGNOSTIC
static float composed_debug[32];
#define DEBUG_VALUE(index,value) (composed_debug[(index)]=(value))
#else
#define DEBUG_VALUE(index,value) do{(void)sizeof(index);(void)sizeof(value);}while(0)
#endif
#ifdef COMPOSED_FRONTEND_BENCHMARK
static volatile float sink;
#endif

static float pow2(float x,int32_t d){bits_t b={.f=x};int32_t e=(int32_t)((b.u>>23)&255)+d;if(e<=0)return 0;if(e>=255){b.u=0x7f800000;return b.f;}b.u=(b.u&0x807fffff)|((uint32_t)e<<23);return b.f;}
#ifdef COMPOSED_PEDAL_HOST
static void sinhcosh(float x,float*s,float*c){if(x>80)x=80;if(x< -80)x=-80;*s=sinhf(x);*c=coshf(x);}
#else
static void sinhcosh(float x,float*s,float*c){const float il=1.44269504089F,l=.69314718056F,k=1073741824.0F,ik=9.313225746e-10F;if(x>80)x=80;if(x< -80)x=-80;float y=x*il;int32_t e=(int32_t)(y+(y>=0?.5F:-.5F));float r=x-(float)e*l;CORDIC->WDATA=(uint32_t)(int32_t)(r*k);int32_t cq=(int32_t)CORDIC->RDATA,sq=(int32_t)CORDIC->RDATA;float p=pow2((float)(cq+sq)*ik,e),n=pow2((float)(cq-sq)*ik,-e);*s=.5F*(p-n);*c=.5F*(p+n);}
#endif
static void cached_exp(float s,float c,float*i,float*d){float e=s+c;*i=IS*(e-1);*d=IS*e/FS;}
#if !defined(COMPOSED_Q1_DIRECT_EXP) || defined(COMPOSED_INSTANT_INTERFACE)
static void cached_exp_sat(float s,float c,float saturation,float*i,float*d){float e=s+c;*i=saturation*(e-1);*d=saturation*e/FS;}
#endif
#ifdef COMPOSED_Q1_DIRECT_EXP
static float exact_exp(float q,float scale){float s,c;sinhcosh(q/scale,&s,&c);return s+c;}
static void cached_exp_direct(float e,float saturation,float*i,float*d){*i=saturation*(e-1);*d=saturation*e/FS;}
static void update_exp_cache(float oldq,float newq,float scale,float*e,uint32_t*fallbacks){float x=(newq-oldq)/scale,ax=__builtin_fabsf(x);if(ax>COMPOSED_Q1_DIRECT_EXACT_THRESHOLD){*e=exact_exp(newq,scale);(*fallbacks)++;return;}float x2=x*x,sd=x+x*x2*(1.0F/6.0F),cd=1.0F+.5F*x2;if(ax>.25F){float x4=x2*x2;sd+=x*x4*(1.0F/120.0F);cd+=x4*(1.0F/24.0F);}*e*=cd+sd;}
#endif
static void cached_diode(float s,float c,float*i,float*d){*i=DIS2*s;*d=DIS2*c/DS;}
static void update_cache(float oldq,float newq,float scale,float*sv,float*cv,uint32_t*fallbacks){float x=(newq-oldq)/scale,ax=__builtin_fabsf(x);if(ax>.5F){sinhcosh(newq/scale,sv,cv);(*fallbacks)++;return;}float x2=x*x,sd=x+x*x2*(1.0F/6.0F),cd=1.0F+.5F*x2;if(ax>.25F){float x4=x2*x2;sd+=x*x4*(1.0F/120.0F);cd+=x4*(1.0F/24.0F);}float os=*sv,oc=*cv;*sv=os*cd+oc*sd;*cv=oc*cd+os*sd;}
static void refresh(state_t*s){sinhcosh(s->aq[0]/FS,&s->av[0],&s->ac[0]);sinhcosh(s->aq[1]/FS,&s->av[1],&s->ac[1]);sinhcosh(s->aq[2]/DS,&s->av[2],&s->ac[2]);sinhcosh(s->bq/DS,&s->bv,&s->bc);
#ifdef COMPOSED_Q1_DIRECT_EXP
 s->se[0]=exact_exp(s->sq[0],FS);s->se[1]=exact_exp(s->sq[1],FS);
#else
 sinhcosh(s->sq[0]/FS,&s->sv[0],&s->sc[0]);sinhcosh(s->sq[1]/FS,&s->sv[1],&s->sc[1]);
#endif
}
#if !defined(COMPOSED_PEDAL_HOST_FULL_REFRESH) && !defined(COMPOSED_SLOW_FULL_REFRESH)
static void refresh_one(state_t*s,uint32_t which){switch(which){case 0:sinhcosh(s->aq[0]/FS,&s->av[0],&s->ac[0]);break;case 1:sinhcosh(s->aq[1]/FS,&s->av[1],&s->ac[1]);break;case 2:sinhcosh(s->aq[2]/DS,&s->av[2],&s->ac[2]);break;case 3:
#ifdef COMPOSED_Q1_DIRECT_EXP
 s->se[0]=exact_exp(s->sq[0],FS);
#else
 sinhcosh(s->sq[0]/FS,&s->sv[0],&s->sc[0]);
#endif
 break;default:
#ifdef COMPOSED_Q1_DIRECT_EXP
 s->se[1]=exact_exp(s->sq[1],FS);
#else
 sinhcosh(s->sq[1]/FS,&s->sv[1],&s->sc[1]);
#endif
 break;}}
#endif
static void solve3(float m[3][3],float r[3],float x[3]){float a=m[0][0],b=m[0][1],c=m[0][2],d=m[1][0],e=m[1][1],f=m[1][2],g=m[2][0],h=m[2][1],i=m[2][2];float c0=e*i-f*h,c1=f*g-d*i,c2=d*h-e*g,z=1/(a*c0+b*c1+c*c2);x[0]=z*(c0*r[0]+(c*h-b*i)*r[1]+(b*f-c*e)*r[2]);x[1]=z*(c1*r[0]+(a*i-c*g)*r[1]+(c*d-a*f)*r[2]);x[2]=z*(c2*r[0]+(b*g-a*h)*r[1]+(a*e-b*d)*r[2]);}
static float dot(const float*a,const float*b,uint32_t n){float v=0;for(uint32_t k=0;k<n;k++)v+=a[k]*b[k];return v;}
static void history(const float*state,const float*older,float*out,uint32_t n){
#ifdef COMPOSED_GENERALIZED_ALPHA02
 for(uint32_t k=0;k<n;k++)out[k]=state[k]+(5.0F/21.0F)*older[k];
#elif defined(COMPOSED_BDF2)
 for(uint32_t k=0;k<n;k++)out[k]=(4.0F*state[k]-older[k])*(1.0F/3.0F);
#else
 (void)older;for(uint32_t k=0;k<n;k++)out[k]=state[k];
#endif
}
static void accept_state(float*state,float*history_state,const float*stage,uint32_t n){
#ifdef COMPOSED_GENERALIZED_ALPHA02
 for(uint32_t k=0;k<n;k++){float delta=1.2F*(stage[k]-state[k]);state[k]+=delta;history_state[k]=1.2F*delta-.2F*history_state[k];}
#else
 for(uint32_t k=0;k<n;k++){history_state[k]=state[k];state[k]=stage[k];}
#endif
}
/* Correctly matched Q2 is experimental: the full-chain SPICE error increased. */
#if (defined(COMPOSED_BDF2) || defined(COMPOSED_GENERALIZED_ALPHA02)) && defined(COMPOSED_Q2_MATCHED)
#include "composed_q2_polynomial.h"
#else
static float q2_analytic_predictor(float linear_q){
 const float inv_ds=1.0F/DS;float t=__builtin_fabsf(linear_q)*inv_ds,z,y;
 if(t<4.0F){z=.5F*t-1.0F;y=(((((-.00194939668F*z-.00618773419F)*z-.0130628288F)*z-.0194541700F)*z+1.97923589F)*z+1.98988569F);}
 else if(t<8.0F){z=.5F*t-3.0F;y=(((((.0016144237F*z+.0242019072F)*z-.0230315961F)*z-.287787288F)*z+1.4428699F)*z+5.61377478F);}
 else if(t<12.0F){z=.5F*t-5.0F;y=(((((.000428220927F*z-.00462351367F)*z+.0252768416F)*z-.115978062F)*z+.569298327F)*z+7.48695326F);}
 else if(t<24.0F){z=(t-18.0F)*(1.0F/6.0F);y=(((((.00872084219F*z-.0228137802F)*z+.0511515848F)*z-.154222712F)*z+.587535262F)*z+8.78609085F);}
 else {z=(t-39.0F)*(1.0F/15.0F);y=(((((.00644029491F*z-.0164690986F)*z+.0373023376F)*z-.119219266F)*z+.499001712F)*z+9.93493462F);}
 return __builtin_copysignf(y*DS,linear_q);
}
#endif
static void slow_stage(state_t*s,float port){
 s->last_port=port;
 float hs[4];history(s->ss,
#if defined(COMPOSED_BDF2) || defined(COMPOSED_GENERALIZED_ALPHA02)
 s->ss_old,
#else
 s->ss,
#endif
 hs,4);
 float lq[2],cur[2],der[2],saved_sq[2]={s->sq[0],s->sq[1]},saved_sv[2]={s->sv[0],s->sv[1]},saved_sc[2]={s->sc[0],s->sc[1]},saved_ss[4]={s->ss[0],s->ss[1],s->ss[2],s->ss[3]},saved_output=s->output;uint32_t safe=1;
#ifdef COMPOSED_Q1_DIRECT_EXP
 float saved_se[2]={s->se[0],s->se[1]};
#endif
 DEBUG_VALUE(0,port);DEBUG_VALUE(1,s->sq[0]);DEBUG_VALUE(2,s->sq[1]);
 for(uint32_t j=0;j<4;j++)DEBUG_VALUE(3+j,hs[j]);
 for(uint32_t j=0;j<2;j++)lq[j]=composed_slow_active_bias[j]+composed_slow_active_port[j]*port+dot(composed_slow_active_state[j],hs,4);
 DEBUG_VALUE(7,lq[0]);DEBUG_VALUE(8,lq[1]);
 for(uint32_t pass=0;pass<2;pass++){
#ifdef COMPOSED_Q1_DIRECT_EXP
  cached_exp_direct(s->se[0],IS,&cur[0],&der[0]);cached_exp_direct(s->se[1],12.0e-15F,&cur[1],&der[1]);
#else
  cached_exp_sat(s->sv[0],s->sc[0],IS,&cur[0],&der[0]);cached_exp_sat(s->sv[1],s->sc[1],12.0e-15F,&cur[1],&der[1]);
#endif
  float r0=s->sq[0]-lq[0]+composed_slow_active_influence[0][0]*cur[0]+composed_slow_active_influence[0][1]*cur[1];
  float r1=s->sq[1]-lq[1]+composed_slow_active_influence[1][0]*cur[0]+composed_slow_active_influence[1][1]*cur[1];
  float norm=__builtin_fmaxf(__builtin_fabsf(r0),__builtin_fabsf(r1));if(norm<1.0e-6F)break;
  float a=1+composed_slow_active_influence[0][0]*der[0],b=composed_slow_active_influence[0][1]*der[1],c=composed_slow_active_influence[1][0]*der[0],d=1+composed_slow_active_influence[1][1]*der[1],z=1/(a*d-b*c);
  float dx0=z*(d*r0-b*r1),dx1=z*(a*r1-c*r0),old0=s->sq[0],old1=s->sq[1];uint32_t exact_new=0;
  uint32_t o=9U+pass*7U;DEBUG_VALUE(o,r0);DEBUG_VALUE(o+1,r1);DEBUG_VALUE(o+2,norm);DEBUG_VALUE(o+3,a*d-b*c);DEBUG_VALUE(o+4,dx0);DEBUG_VALUE(o+5,dx1);DEBUG_VALUE(o+6,z);
  if(__builtin_fmaxf(__builtin_fabsf(dx0),__builtin_fabsf(dx1))>.25F*FS){
   s->adaptive++;s->slow_adaptive++;exact_new=1;
  }
  s->sq[0]-=dx0;s->sq[1]-=dx1;
  if(exact_new){
#ifdef COMPOSED_Q1_DIRECT_EXP
   uint32_t exact0=__builtin_fabsf(dx0)>COMPOSED_Q1_DIRECT_EXACT_THRESHOLD*FS,exact1=__builtin_fabsf(dx1)>COMPOSED_Q1_DIRECT_EXACT_THRESHOLD*FS;
   if(exact0){s->se[0]=exact_exp(s->sq[0],FS);s->q1_exact[0]++;}else update_exp_cache(old0,s->sq[0],FS,&s->se[0],&s->fallbacks);
   if(exact1){s->se[1]=exact_exp(s->sq[1],FS);s->q1_exact[1]++;}else update_exp_cache(old1,s->sq[1],FS,&s->se[1],&s->fallbacks);
#else
   sinhcosh(s->sq[0]/FS,&s->sv[0],&s->sc[0]);sinhcosh(s->sq[1]/FS,&s->sv[1],&s->sc[1]);
#endif
  }
  else {
#ifdef COMPOSED_Q1_DIRECT_EXP
   update_exp_cache(old0,s->sq[0],FS,&s->se[0],&s->fallbacks);update_exp_cache(old1,s->sq[1],FS,&s->se[1],&s->fallbacks);
#else
   update_cache(old0,s->sq[0],FS,&s->sv[0],&s->sc[0],&s->fallbacks);update_cache(old1,s->sq[1],FS,&s->sv[1],&s->sc[1],&s->fallbacks);
#endif
  }
 }
 if(!safe){s->sq[0]=saved_sq[0];s->sq[1]=saved_sq[1];s->sv[0]=saved_sv[0];s->sv[1]=saved_sv[1];s->sc[0]=saved_sc[0];s->sc[1]=saved_sc[1];
#ifdef COMPOSED_Q1_DIRECT_EXP
 s->se[0]=saved_se[0];s->se[1]=saved_se[1];
#endif
 {for(uint32_t j=0;j<4;j++)s->ss[j]=saved_ss[j];s->output=saved_output;s->holds++;return;}}
#ifdef COMPOSED_Q1_DIRECT_EXP
 cached_exp_direct(s->se[0],IS,&cur[0],&der[0]);cached_exp_direct(s->se[1],12.0e-15F,&cur[1],&der[1]);
#else
 cached_exp_sat(s->sv[0],s->sc[0],IS,&cur[0],&der[0]);cached_exp_sat(s->sv[1],s->sc[1],12.0e-15F,&cur[1],&der[1]);
#endif
 float next[4];for(uint32_t j=0;j<4;j++)next[j]=composed_slow_state_bias[j]+composed_slow_state_port[j]*port+dot(composed_slow_state_transition[j],hs,4)-dot(composed_slow_state_active[j],cur,2);
 float stage_output=composed_slow_output_bias[0]+composed_slow_output_port[0]*port+dot(composed_slow_output_state,hs,4)-dot(composed_slow_output_active,cur,2);
 DEBUG_VALUE(23,s->sq[0]);DEBUG_VALUE(24,s->sq[1]);DEBUG_VALUE(25,cur[0]);DEBUG_VALUE(26,cur[1]);DEBUG_VALUE(27,next[0]);DEBUG_VALUE(28,next[1]);DEBUG_VALUE(29,next[2]);DEBUG_VALUE(30,next[3]);DEBUG_VALUE(31,stage_output);
#ifdef COMPOSED_TWO_PORT_CORRECTION
 float port_current=composed_slow_current_bias[0]+composed_slow_current_port[0]*port+dot(composed_slow_current_state,hs,4)+dot(composed_slow_current_active,cur,2);
 float next_offset=port_current-composed_port[1]*port;
 /* Токовая поправка запаздывает на один отсчёт; половинное обновление гасит
    моду около Найквиста и не добавляет нового нелинейного решения. */
 s->port_offset+=COMPOSED_PORT_RELAX*(next_offset-s->port_offset);
#endif
#ifdef COMPOSED_GENERALIZED_ALPHA02
 s->output+=1.2F*(stage_output-s->output);
#else
 s->output=stage_output;
#endif
#if defined(COMPOSED_BDF2) || defined(COMPOSED_GENERALIZED_ALPHA02)
 accept_state(s->ss,s->ss_old,next,4);
#else
 for(uint32_t j=0;j<4;j++)s->ss[j]=next[j];
#endif
}
#ifdef COMPOSED_INSTANT_INTERFACE
static void slow_port_linearization(state_t*s,float port,float*port_current,float*port_slope){
 float hs[4];history(s->ss,s->ss_old,hs,4);
 float q[2]={s->sq[0],s->sq[1]},cur[2],der[2],lq[2];
 for(uint32_t j=0;j<2;j++)lq[j]=composed_slow_active_bias[j]+composed_slow_active_port[j]*port+dot(composed_slow_active_state[j],hs,4);
 float a=1,b=0,c=0,d=1,z=1;
 for(uint32_t pass=0;pass<3;pass++){
  float sv0,cv0,sv1,cv1;sinhcosh(q[0]/FS,&sv0,&cv0);sinhcosh(q[1]/FS,&sv1,&cv1);
  cached_exp_sat(sv0,cv0,IS,&cur[0],&der[0]);cached_exp_sat(sv1,cv1,12.0e-15F,&cur[1],&der[1]);
  float r0=q[0]-lq[0]+composed_slow_active_influence[0][0]*cur[0]+composed_slow_active_influence[0][1]*cur[1];
  float r1=q[1]-lq[1]+composed_slow_active_influence[1][0]*cur[0]+composed_slow_active_influence[1][1]*cur[1];
  a=1+composed_slow_active_influence[0][0]*der[0];b=composed_slow_active_influence[0][1]*der[1];c=composed_slow_active_influence[1][0]*der[0];d=1+composed_slow_active_influence[1][1]*der[1];z=1/(a*d-b*c);
  q[0]-=z*(d*r0-b*r1);q[1]-=z*(a*r1-c*r0);
 }
 float sv0,cv0,sv1,cv1;sinhcosh(q[0]/FS,&sv0,&cv0);sinhcosh(q[1]/FS,&sv1,&cv1);
 cached_exp_sat(sv0,cv0,IS,&cur[0],&der[0]);cached_exp_sat(sv1,cv1,12.0e-15F,&cur[1],&der[1]);
 a=1+composed_slow_active_influence[0][0]*der[0];b=composed_slow_active_influence[0][1]*der[1];c=composed_slow_active_influence[1][0]*der[0];d=1+composed_slow_active_influence[1][1]*der[1];z=1/(a*d-b*c);
 float qp0=z*(d*composed_slow_active_port[0]-b*composed_slow_active_port[1]);
 float qp1=z*(a*composed_slow_active_port[1]-c*composed_slow_active_port[0]);
 *port_current=composed_slow_current_bias[0]+composed_slow_current_port[0]*port+dot(composed_slow_current_state,hs,4)+dot(composed_slow_current_active,cur,2);
 *port_slope=composed_slow_current_port[0]+composed_slow_current_active[0]*der[0]*qp0+composed_slow_current_active[1]*der[1]*qp1;
}
#endif
static void solve_q2(state_t*s,float linear_q,float*current,float*derivative){
 const float influence=composed_b_influence[0][0];
 if(linear_q<s->b_linear_min)s->b_linear_min=linear_q;
 if(linear_q>s->b_linear_max)s->b_linear_max=linear_q;
 s->b_linear=linear_q;
 if(__builtin_fabsf(linear_q)<=54.0F*DS){
  s->bq=q2_analytic_predictor(linear_q);*current=(linear_q-s->bq)/influence;
  s->bv=*current/DIS2;s->bc=__builtin_sqrtf(1.0F+s->bv*s->bv);*derivative=DIS2*s->bc/DS;return;
 }
 s->adaptive++;s->q2_bracketed++;float low=linear_q<0?linear_q:0,high=linear_q>0?linear_q:0,q=s->bq;if(q<low)q=low;if(q>high)q=high;
 for(uint32_t pass=0;pass<20;pass++){
  float sv,cv;sinhcosh(q/DS,&sv,&cv);float ci,cd;cached_diode(sv,cv,&ci,&cd);float f=q-linear_q+influence*ci;
  if(f<0)low=q;else high=q;
  if(__builtin_fabsf(f)<1.0e-6F){s->bq=q;s->bv=sv;s->bc=cv;*current=ci;*derivative=cd;return;}
  float candidate=q-f/(1+influence*cd);
  if(candidate<=low||candidate>=high||((((bits_t){.f=candidate}.u>>23)&255U)==255U))candidate=.5F*(low+high);
  q=candidate;
 }
 sinhcosh(q/DS,&s->bv,&s->bc);s->bq=q;cached_diode(s->bv,s->bc,current,derivative);
}

__attribute__((noinline)) static void step(state_t*s,float input){
#if !defined(COMPOSED_PEDAL_HOST_FULL_REFRESH) && !defined(COMPOSED_SLOW_FULL_REFRESH)
 uint32_t fallback_before=s->fallbacks;
#endif
 float ah[7],bh[2];history(s->as,s->as_old,ah,7);history(s->bs,s->bs_old,bh,2);
#ifdef COMPOSED_GENERALIZED_ALPHA02
 float stage_input=s->previous_input+(5.0F/6.0F)*(input-s->previous_input);
 float predicted=s->base+(5.0F/6.0F)*(s->base-s->base_old);
#else
 float stage_input=input,predicted=2*s->base-s->base_old;
#endif
 float lq[3],cur[3],der[3],m[3][3],r[3],dx[3];
 for(uint32_t j=0;j<3;j++)lq[j]=composed_a_q_bias[j]+composed_a_q_input[j]*stage_input+composed_a_q_boundary[j]*predicted+dot(composed_a_q_state[j],ah,7);
 cached_exp(s->av[0],s->ac[0],&cur[0],&der[0]);cached_exp(s->av[1],s->ac[1],&cur[1],&der[1]);cached_diode(s->av[2],s->ac[2],&cur[2],&der[2]);
 for(uint32_t j=0;j<3;j++){r[j]=s->aq[j]-lq[j];for(uint32_t k=0;k<3;k++){r[j]+=composed_a_influence[j][k]*cur[k];m[j][k]=(j==k)+composed_a_influence[j][k]*der[k];}}solve3(m,r,dx);
 float olda[3]={s->aq[0],s->aq[1],s->aq[2]};for(uint32_t j=0;j<3;j++)s->aq[j]-=dx[j];
 update_cache(olda[0],s->aq[0],FS,&s->av[0],&s->ac[0],&s->fallbacks);update_cache(olda[1],s->aq[1],FS,&s->av[1],&s->ac[1],&s->fallbacks);update_cache(olda[2],s->aq[2],DS,&s->av[2],&s->ac[2],&s->fallbacks);
 cached_exp(s->av[0],s->ac[0],&cur[0],&der[0]);cached_exp(s->av[1],s->ac[1],&cur[1],&der[1]);cached_diode(s->av[2],s->ac[2],&cur[2],&der[2]);
 float drive=composed_a_drive_bias[0]+composed_a_drive_input[0]*stage_input+composed_a_drive_boundary[0]*predicted+dot(composed_a_drive_state,ah,7)-dot(composed_a_drive_active,cur,3),nexta[7];
 if((((bits_t){.f=drive}.u>>23)&255U)==255U)s->fault=1U;
 for(uint32_t j=0;j<7;j++)nexta[j]=composed_a_state_bias[j]+composed_a_state_input[j]*stage_input+composed_a_state_boundary[j]*predicted+dot(composed_a_state_transition[j],ah,7)-dot(composed_a_state_active[j],cur,3);
 float lqb=composed_b_q_bias[0]+composed_b_q_input[0]*drive+composed_b_q_port[0]*s->port_offset+dot(composed_b_q_state[0],bh,2),ci,cd;
 solve_q2(s,lqb,&ci,&cd);
 float base=composed_b_base_bias[0]+composed_b_base_input[0]*drive+composed_b_base_port[0]*s->port_offset+dot(composed_b_base_state,bh,2)-composed_b_base_active[0]*ci;
 float collector=composed_b_collector_bias[0]+composed_b_collector_input[0]*drive+composed_b_collector_port[0]*s->port_offset+dot(composed_b_collector_state,bh,2)-composed_b_collector_active[0]*ci;
 if(((((bits_t){.f=base}.u>>23)&255U)==255U)||((((bits_t){.f=collector}.u>>23)&255U)==255U))s->fault=2U;
 float nextb[2];for(uint32_t j=0;j<2;j++)nextb[j]=composed_b_state_bias[j]+composed_b_state_input[j]*drive+composed_b_state_port[j]*s->port_offset+dot(composed_b_state_transition[j],bh,2)-composed_b_state_active[j][0]*ci;
#ifdef COMPOSED_INSTANT_INTERFACE
 float qb2[3],dib2[3];solve3(m,(float[3]){composed_a_q_boundary[0],composed_a_q_boundary[1],composed_a_q_boundary[2]},qb2);
 for(uint32_t j=0;j<3;j++)dib2[j]=der[j]*qb2[j];
 float drive_x=composed_a_drive_boundary[0]-dot(composed_a_drive_active,dib2,3);
 float q_drive=composed_b_q_input[0]/(1+composed_b_influence[0][0]*cd),q_port=composed_b_q_port[0]/(1+composed_b_influence[0][0]*cd);
 float ci_drive=cd*q_drive,ci_port=cd*q_port;
 float base_drive=composed_b_base_input[0]-composed_b_base_active[0]*ci_drive;
 float base_port=composed_b_base_port[0]-composed_b_base_active[0]*ci_port;
 float collector_drive=composed_b_collector_input[0]-composed_b_collector_active[0]*ci_drive;
 float collector_port=composed_b_collector_port[0]-composed_b_collector_active[0]*ci_port;
 float slow_i,slow_g;slow_port_linearization(s,collector,&slow_i,&slow_g);
 float load_delta=slow_g-composed_port[1],bx=base_drive*drive_x,vx=collector_drive*drive_x;
 float r0=base-predicted,r1=s->port_offset-(slow_i-composed_port[1]*collector);
 float j00=bx-1,j01=base_port,j10=-load_delta*vx,j11=1-load_delta*collector_port;
 float determinant=j00*j11-j01*j10;
 if(__builtin_fabsf(determinant)>1.0e-6F){
  float dx=(-r0*j11+j01*r1)/determinant,dp=(-j00*r1+j10*r0)/determinant;
  if(dx>.02F)dx=.02F;else if(dx<-.02F)dx=-.02F;if(dp>5e-5F)dp=5e-5F;else if(dp< -5e-5F)dp=-5e-5F;
  for(uint32_t j=0;j<3;j++){s->aq[j]+=qb2[j]*dx;cur[j]+=dib2[j]*dx;}
  drive+=drive_x*dx;for(uint32_t j=0;j<7;j++)nexta[j]+=(composed_a_state_boundary[j]-dot(composed_a_state_active[j],dib2,3))*dx;
  float dd=drive_x*dx;s->bq+=q_drive*dd+q_port*dp;ci+=ci_drive*dd+ci_port*dp;s->port_offset+=dp;
  base+=base_drive*dd+base_port*dp;collector+=collector_drive*dd+collector_port*dp;
  for(uint32_t j=0;j<2;j++)nextb[j]+=(composed_b_state_input[j]-composed_b_state_active[j][0]*ci_drive)*dd+(composed_b_state_port[j]-composed_b_state_active[j][0]*ci_port)*dp;
 }
#endif
#ifdef COMPOSED_BOUNDARY_CORRECTION
 /* Одна скалярная поправка Ньютона для разрезанной обратной связи через R12.
    Используем уже посчитанные матрицы Якоби: повторно решать нелинейные
    блоки не требуется. */
 float qb[3],dib[3],drive_b,db_dd,dci_dd,base_dd,loop_gain,den,delta;
 solve3(m,(float[3]){composed_a_q_boundary[0],composed_a_q_boundary[1],composed_a_q_boundary[2]},qb);
 for(uint32_t j=0;j<3;j++)dib[j]=der[j]*qb[j];
 drive_b=composed_a_drive_boundary[0]-dot(composed_a_drive_active,dib,3);
 db_dd=composed_b_q_input[0]/(1.0F+composed_b_influence[0][0]*cd);
 dci_dd=cd*db_dd;
 base_dd=composed_b_base_input[0]-composed_b_base_active[0]*dci_dd;
 loop_gain=base_dd*drive_b;den=1.0F-loop_gain;
 if(__builtin_fabsf(den)>0.125F){
  delta=.25F*(base-predicted)/den;
  /* На сильном переходе не выпускаем поправку из области линеаризации. */
  if(delta>.01F)delta=.01F;else if(delta<-.01F)delta=-.01F;
  for(uint32_t j=0;j<3;j++){s->aq[j]+=qb[j]*delta;cur[j]+=dib[j]*delta;}
  drive+=drive_b*delta;
  for(uint32_t j=0;j<7;j++)nexta[j]+=(composed_a_state_boundary[j]-dot(composed_a_state_active[j],dib,3))*delta;
  float drive_delta=drive_b*delta;
  s->bq+=db_dd*drive_delta;ci+=dci_dd*drive_delta;
  base+=loop_gain*delta;
  collector+=(composed_b_collector_input[0]-composed_b_collector_active[0]*dci_dd)*drive_delta;
  for(uint32_t j=0;j<2;j++)nextb[j]+=(composed_b_state_input[j]-composed_b_state_active[j][0]*dci_dd)*drive_delta;
 }
#endif
 accept_state(s->as,s->as_old,nexta,7);accept_state(s->bs,s->bs_old,nextb,2);
#ifdef COMPOSED_GENERALIZED_ALPHA02
 float next_base=s->base+1.2F*(base-s->base);s->base_old=s->base;s->base=next_base;
 slow_stage(s,collector);s->previous_input=input;
#else
 s->base_old=s->base;s->base=base;slow_stage(s,collector);
#endif
 if((((bits_t){.f=s->output}.u>>23)&255U)==255U){s->fault=3U;}
 s->index++;
#ifdef COMPOSED_PEDAL_HOST_FULL_REFRESH
 if((s->index&15U)==0U)refresh(s);
#elif defined(COMPOSED_SLOW_FULL_REFRESH)
 if((s->index&15U)==0U){sinhcosh(s->sq[0]/FS,&s->sv[0],&s->sc[0]);sinhcosh(s->sq[1]/FS,&s->sv[1],&s->sc[1]);}
#else
 if((s->index&15U)==0U&&s->fallbacks==fallback_before)refresh_one(s,(s->index>>4U)%5U);
#endif
}
static void reset(state_t*s){for(uint32_t j=0;j<7;j++){s->as[j]=composed_a_dc_state[j];s->as_old[j]=
#ifdef COMPOSED_GENERALIZED_ALPHA02
0.0F;
#else
composed_a_dc_state[j];
#endif
}for(uint32_t j=0;j<3;j++)s->aq[j]=composed_a_dc_q[j];for(uint32_t j=0;j<2;j++){s->bs[j]=composed_b_dc_state[j];s->bs_old[j]=
#ifdef COMPOSED_GENERALIZED_ALPHA02
0.0F;
#else
composed_b_dc_state[j];
#endif
}s->bq=composed_b_dc_q[0];s->b_linear=0;s->b_linear_min=0;s->b_linear_max=0;for(uint32_t j=0;j<4;j++){
#if defined(COMPOSED_BDF2) || defined(COMPOSED_GENERALIZED_ALPHA02)
#ifdef COMPOSED_GENERALIZED_ALPHA02
 s->ss_old[j]=0.0F;
#else
 s->ss_old[j]=composed_slow_dc_state[j];
#endif
#endif
 s->ss[j]=composed_slow_dc_state[j];}for(uint32_t j=0;j<2;j++){s->sq[j]=composed_slow_dc_q[j];s->q1_exact[j]=0;}s->base=s->base_old=.695552931F;s->output=0;s->last_port=0;s->port_offset=composed_port[0];s->previous_input=0;s->index=0;s->fallbacks=0;s->adaptive=0;s->slow_adaptive=0;s->q2_bracketed=0;s->holds=0;s->fault=0;refresh(s);}

#if defined(COMPOSED_PEDAL_RUNTIME) || defined(COMPOSED_PEDAL_HOST)
static state_t runtime_state;

void composed_pedal_init(void)
{
#ifndef COMPOSED_PEDAL_HOST
 RCC->AHB1ENR|=RCC_AHB1ENR_CORDICEN;
 RCC->AHB1RSTR|=RCC_AHB1RSTR_CORDICRST;
 RCC->AHB1RSTR&=~RCC_AHB1RSTR_CORDICRST;
 CORDIC->CSR=CORDIC_CSR_FUNC_2|CORDIC_CSR_FUNC_0|CORDIC_CSR_PRECISION_2|CORDIC_CSR_PRECISION_1|CORDIC_CSR_SCALE_0|CORDIC_CSR_NRES;
#endif
 reset(&runtime_state);
}

float composed_pedal_process_sample(float input)
{
 step(&runtime_state,input);
 return runtime_state.output;
}
#ifdef COMPOSED_PEDAL_HOST_DIAGNOSTIC
void composed_pedal_debug(float *values)
{
 for(uint32_t j=0;j<32;j++)values[j]=composed_debug[j];
}
#endif

uint32_t composed_pedal_fallback_count(void)
{
 return runtime_state.fallbacks;
}

uint32_t composed_pedal_adaptive_count(void)
{
 return runtime_state.adaptive;
}

uint32_t composed_pedal_fault_stage(void)
{
 return runtime_state.fault;
}

uint32_t composed_pedal_hold_count(void)
{
 return runtime_state.holds;
}

float composed_pedal_last_port(void)
{
 return runtime_state.last_port;
}

float composed_pedal_q2_linear_min(void)
{
 return runtime_state.b_linear_min;
}

float composed_pedal_q2_linear_max(void)
{
 return runtime_state.b_linear_max;
}
#endif

#ifdef COMPOSED_FRONTEND_BENCHMARK
static uint32_t count(void){__asm volatile("":::"memory");uint32_t x=DWT->CYCCNT;__asm volatile("":::"memory");return x;}
static void clockinit(void){RCC->APB1ENR1|=RCC_APB1ENR1_PWREN;PWR->CR5&=~PWR_CR5_R1MODE;PWR->CR1=(PWR->CR1&~PWR_CR1_VOS)|PWR_CR1_VOS_0;while(PWR->SR2&PWR_SR2_VOSF){}FLASH->ACR=FLASH_ACR_PRFTEN|FLASH_ACR_ICEN|FLASH_ACR_DCEN|FLASH_ACR_LATENCY_4WS;RCC->CR|=RCC_CR_HSION;while(!(RCC->CR&RCC_CR_HSIRDY)){}RCC->CR&=~RCC_CR_PLLON;while(RCC->CR&RCC_CR_PLLRDY){}RCC->PLLCFGR=RCC_PLLCFGR_PLLSRC_HSI|(3UL<<RCC_PLLCFGR_PLLM_Pos)|(85UL<<RCC_PLLCFGR_PLLN_Pos)|RCC_PLLCFGR_PLLREN;RCC->CR|=RCC_CR_PLLON;while(!(RCC->CR&RCC_CR_PLLRDY)){}RCC->CFGR=(RCC->CFGR&~(RCC_CFGR_SW|RCC_CFGR_HPRE|RCC_CFGR_PPRE1|RCC_CFGR_PPRE2))|RCC_CFGR_SW_PLL;while((RCC->CFGR&RCC_CFGR_SWS)!=RCC_CFGR_SWS_PLL){}SystemCoreClock=170000000;}
static void uartinit(void){RCC->AHB2ENR|=RCC_AHB2ENR_GPIOAEN;RCC->APB1ENR1|=RCC_APB1ENR1_USART2EN;GPIOA->MODER=(GPIOA->MODER&~(3UL<<4))|(2UL<<4);GPIOA->AFR[0]=(GPIOA->AFR[0]&~(15UL<<8))|(7UL<<8);USART2->BRR=(SystemCoreClock+57600)/115200;USART2->CR1=USART_CR1_TE|USART_CR1_UE;while(!(USART2->ISR&USART_ISR_TEACK)){} }
static void ch(char c){while(!(USART2->ISR&USART_ISR_TXE)){}USART2->TDR=(uint8_t)c;}static void txt(const char*x){while(*x)ch(*x++);}static void num(uint32_t v){char d[10];uint32_t n=0;do{d[n++]=(char)('0'+v%10);v/=10;}while(v);while(n)ch(d[--n]);}static void report(const char*n,uint32_t v){txt(n);ch('=');num(v);txt("\r\n");}
#ifdef COMPOSED_BLOCK_BENCHMARK
extern const uint8_t composed_chord_wav_start[];
extern const uint8_t composed_chord_wav_end[];
static uint32_t le32(const uint8_t*p){return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);}
int main(void){
 clockinit();uartinit();CoreDebug->DEMCR|=CoreDebug_DEMCR_TRCENA_Msk;DWT->CYCCNT=0;DWT->CTRL|=DWT_CTRL_CYCCNTENA_Msk;RCC->AHB1ENR|=RCC_AHB1ENR_CORDICEN;RCC->AHB1RSTR|=RCC_AHB1RSTR_CORDICRST;RCC->AHB1RSTR&=~RCC_AHB1RSTR_CORDICRST;CORDIC->CSR=CORDIC_CSR_FUNC_2|CORDIC_CSR_FUNC_0|CORDIC_CSR_PRECISION_2|CORDIC_CSR_PRECISION_1|CORDIC_CSR_SCALE_0|CORDIC_CSR_NRES;
 const uint32_t bytes=(uint32_t)(composed_chord_wav_end-composed_chord_wav_start);const uint32_t samples=le32(composed_chord_wav_start+40)/2U;const int16_t*pcm=(const int16_t*)(composed_chord_wav_start+44);int64_t sum=0;float peak=1;
 for(uint32_t n=0;n<samples;n++){sum+=pcm[n];}float mean=(float)sum/(float)samples;for(uint32_t n=0;n<samples;n++){float a=__builtin_fabsf((float)pcm[n]-mean);if(a>peak)peak=a;}
 const uint32_t block_samples=32U;state_t s;reset(&s);uint64_t total=0;uint32_t maximum=0,overruns=0,blocks=0,bad=0,max_block_slow=0;const uint32_t budget=(SystemCoreClock/48000U)*block_samples;
 for(uint32_t base=0;base+block_samples<=samples;base+=block_samples){uint32_t slow_before=s.slow_adaptive,a=count();for(uint32_t n=0;n<block_samples;n++){float input=((float)pcm[base+n]-mean)*(.1F/peak);step(&s,input);float code=2048.0F+s.output*(4095.0F/3.3F);if(code<0)code=0;if(code>4095)code=4095;sink=code;}uint32_t elapsed=count()-a,block_slow=s.slow_adaptive-slow_before;total+=elapsed;if(elapsed>maximum)maximum=elapsed;if(elapsed>budget)overruns++;if(block_slow>max_block_slow)max_block_slow=block_slow;if((((bits_t){.f=s.output}.u>>23)&255U)==255U)bad++;blocks++;}
 uint32_t average=(uint32_t)(total/blocks);for(;;){txt("COMPOSED_FULL_CHORD_BLOCK32_REALTIME_V1\r\n");report("wav_bytes",bytes);report("samples",samples);report("block_samples",block_samples);report("blocks",blocks);report("block_budget_cycles",budget);report("average_block_cycles",average);report("maximum_block_cycles",maximum);report("deadline_overruns",overruns);report("nonfinite_blocks",bad);report("fallback_count",s.fallbacks);report("adaptive_count",s.adaptive);report("slow_large_step_count",s.slow_adaptive);report("q1_exact_0_count",s.q1_exact[0]);report("q1_exact_1_count",s.q1_exact[1]);report("max_block_slow_large_steps",max_block_slow);report("q2_bracketed_count",s.q2_bracketed);report("hold_count",s.holds);report("fault_stage",s.fault);report("final_output_bits",(bits_t){.f=s.output}.u);txt("END\r\n");for(volatile uint32_t d=0;d<20000000;d++){} }
}
#else
int main(void){clockinit();uartinit();CoreDebug->DEMCR|=CoreDebug_DEMCR_TRCENA_Msk;DWT->CYCCNT=0;DWT->CTRL|=DWT_CTRL_CYCCNTENA_Msk;RCC->AHB1ENR|=RCC_AHB1ENR_CORDICEN;RCC->AHB1RSTR|=RCC_AHB1RSTR_CORDICRST;RCC->AHB1RSTR&=~RCC_AHB1RSTR_CORDICRST;CORDIC->CSR=CORDIC_CSR_FUNC_2|CORDIC_CSR_FUNC_0|CORDIC_CSR_PRECISION_2|CORDIC_CSR_PRECISION_1|CORDIC_CSR_SCALE_0|CORDIC_CSR_NRES;uint64_t total=0;uint32_t maximum=0,bad=0,fallbacks=0,adaptive=0,holds=0;state_t s;for(uint32_t p=0;p<PASSES;p++){reset(&s);for(uint32_t n=1;n<PORT_STREAM_COUNT;n++){uint32_t a=count();step(&s,port_stream_input[n]);uint32_t c=count()-a;total+=c;if(c>maximum)maximum=c;if((((bits_t){.f=s.output}.u>>23)&255U)==255U)bad++;}fallbacks+=s.fallbacks;adaptive+=s.adaptive;holds+=s.holds;}sink=s.output;uint32_t samples=PASSES*(PORT_STREAM_COUNT-1),average=(uint32_t)(total/samples);for(;;){txt("COMPOSED_FULL_1X_ADAPTIVE_V2\r\n");report("core_hz",SystemCoreClock);report("rate_hz",48000);report("budget_cycles",SystemCoreClock/48000);report("average_cycles",average);report("maximum_cycles",maximum);report("nonfinite_count",bad);report("fallback_count",fallbacks);report("adaptive_count",adaptive);report("hold_count",holds);report("final_output_bits",(bits_t){.f=s.output}.u);txt("END\r\n");for(volatile uint32_t d=0;d<20000000;d++){} }}
#endif
#endif
#endif
