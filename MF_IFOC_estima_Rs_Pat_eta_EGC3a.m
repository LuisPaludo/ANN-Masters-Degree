%Simula��o do Motor de indu��o trif�sico montado no laborat�rio do POLITEC.
%Foi utilizado a forma de implementa��o em um inversor real a uma freq de 
%chaveamento de 6kHz.

%ser� estimado a resist�ncia estat�rica usnado a t�cnica MRAC pela
%Pot�ncia ativa
clc;
clear all;
close all;

%Dados do Motor 
P=8;            %polos do motor
J=0.1033;       %inercia do rotor
D=0.05;        %atrito do motor 0.002
rr=0.88;       %resistencia do rotor referido ao lado do estator  
rs=2.03;        %resistencia do estator
Llr= 8.434e-3;     %indutancia de dispers�o do rotor
Lls= 8.434e-3;     %indutancia de dispers�o do estator
LM = 116.101e-3;   %indutancia de magnetiza��o

%Dados para simula��o
dt=5e-6;    %passo de simula��o continuo
tf=25;      %tempo final de simula��o
fs=6e3;     %frequencia de chaveamento dos IGBTs
Ts=1/fs;    %periodo de chaveamento

npsd=ceil(tf/Ts);    %n�mero de chaveamentos durante o tempo de simula��o
npc=ceil(Ts/dt);     %n�mero de ciclos por chaveamento
npsc=npsd*npc;       %n�mero total de pontos da simulacao
tfc=dt*npsc;         %tempo total de simula��o corrigida
Tsc=dt*npc;          %Perido corrigido

t=0:Tsc:(npsd-1)*Tsc;      %vetor de tempo de simula��o total apenas em intervalos de chaveamento

lambda_nominal=0.72;
%TL=33.34*(heaviside(t-10)-heaviside(t-15));    %torque da carga aplicado em 10 s e retirado em 15s
%TL=5*(heaviside(t-10)-heaviside(t-15));    % 5 torque da carga aplicado em 10 s e retirado em 15s
TL=5*(heaviside(t-8)-heaviside(t-20));    % 5 torque da carga aplicado em 10 s e retirado em 15s


%Inicializa��o dos vetores e Matrizes para uma simula��o mais r�pida
A=zeros(2);     %Matriz A no espa�o de estados da parte el�trica
B=zeros(2);     %Matriz B no espa�o de estados da parte el�trica

%
%inicializa as vari�veis
%
Vqs_v=zeros(1,length(t));
Vds_v=zeros(1,length(t));
iqs_v=zeros(1,length(t));
iqs=0;
ids_v=zeros(1,length(t));
ids=0;
iqr_v=zeros(1,length(t));
iqr=0;
idr_v=zeros(1,length(t));
idr=0;
ids_ref_v=zeros(1,length(t));

lambda_qs_v=zeros(1,length(t));
lambda_qs=0;
lambda_ds_v=zeros(1,length(t));
lambda_ds=0;
lambda_qr_v=zeros(1,length(t));
lambda_qr=0;
lambda_dr_v=zeros(1,length(t));
lambda_dr=0;
lambda_dr_est_v=zeros(1,length(t));
lambda_dr_est=0;
wr_v=zeros(1,length(t));
wr=0;
Te_v=zeros(1,length(t));
Te=0;
Pin_v=zeros(1,length(t)); % Pot�ncia ativa sugada da rede
Pin=0;
theta_v = zeros(1,length(t));
theta=0;

idm_est_v=zeros(1,length(t));
idm_est=0;
idm_v=zeros(1,length(t));

eta_est_v=zeros(1,length(t));
eta_ref_v=zeros(1,length(t));

rs_v=zeros(1,length(t));
rs_est_v=zeros(1,length(t));
rs_filt_v=zeros(1,length(t));



%Matriz Mec�nica
Am=[-D/J 0; 1 0];
Bm=[P/(2*J) -P/(2*J); 0 0];

theta_r=1e-5;     %posi��o inicial do rotor (as vezes com 0 n�o converge)
Xmi=[wr;theta_r]; %valor inicial da vari�veis de estado mec�nica


%matriz El�trica
Ls=Lls+LM;
Lr=Llr+LM;
sigma=1-LM^2/(Ls*Lr);
%beta=LM/(sigma*Ls*Lr);
Req=rs+rr*(LM/Lr)^2;
Leq=sigma*Ls;
gamma=Req/Leq;
eta=rr/Lr;
delta=(1-sigma)/sigma;

%d(i_qs)/dt
B(1,1)=1/Leq;    %Vqs
B(1,2)=0;        %Vds

%d(i_ds)/dt
B(2,1)=0;        %Vqs
B(2,2)=1/Leq;    %Vds

%d(iqm)/dt
B(3,1)=0;   %Vqs
B(3,2)=0;   %Vds

%d(idm)/dt
B(4,1)=0;   %Vqs
B(4,2)=0;   %Vds


%Ganho dos controladores PI

%controle de ids
Kp_id=10;
Ki_id=20;
inte_id=0;  %valor inicial do integrador 1
erro_id=0;  %valor inicial do erro1
PI_id=0.01;

%controle de iqs
Kp_iq=10;
Ki_iq=20;
inte_iq=0;  %valor inicial do integrador 2
erro_iq=0;  %valor inicial do erro2
PI_iq=0.01;


%Controle de velocidade
Kp_w=1;%0.2
Ki_w=0.5;
inte_w=0;  %valor inicial do integrador 4
erro_w=0;  %valor inicial do erro4
PI_w=0;


%referencias
%lambda_ref=0.2*t.*heaviside(t)-0.2*(t-1).*heaviside(t-1);  %fluxo de referencia
lambda_ref_v=zeros(1,length(t));  %vetor fluxo de referencia
lambda_ref=lambda_nominal;

%%Dados de Rosembrock
%rosembrock = zeros(1,length(t));
%delta_lambda=0.1; %incremento de fluxo para utiliza��o do m�todo de rosembrock
%t_estabiliza=0.4; %tempo estimado em que a pot�ncia estabiliza ap�s degrau de fluxo
%estabiliza=t_estabiliza;  %contagem tempo de estabilizacao estimado para a pot�ncia de entrada
%P2=-inf; %Ser� suposto que a primeira vez a potencia diminui
%DeltaP=-inf;
%t_inicio_rosembrock=13;

%Dados para LMC
t_inicio_otimo=0; %instante de tempo a partir do qual � iniciado o controlador de efici�ncia
                  %normalmente estava sendo colocado no final da rampa,
                  %embora funcione desde o in�cio



%Refer�ncia de velocidade
ini_rampa=2; %tempo de �nicio da rampa
fim_rampa=6; %tempo final da rampa
w_ref_fisico_v=1000*((t-ini_rampa).*heaviside(t-ini_rampa)-(t-fim_rampa).*heaviside(t-fim_rampa))/(fim_rampa-ini_rampa) -200*heaviside(t-10)+200*heaviside(t-15);    %Velocidade de referencia em r.p.m. inicia em rampa
w_ref_v=(P/2)*2*pi/60*w_ref_fisico_v; %Transforma��o para rad/seg e ainda em termos el�tricos
ids_ref=lambda_ref/LM; %referencia inicial de ids


var_ini=1e-4; %valor inicial das correntes para n�o zerar 
Xi=[var_ini;var_ini;var_ini;var_ini];%[iqs;ids;im;im]
I=eye(4);
Im=eye(2);
w=1e-4; %velocidade inicial do sistema de refer�ncia
Energia=0;

%Dados do controlador MRAC
%constante rot�rica
integraQ=0;
K_eta=0.03; %5e-2%ganho do controlador integral de adapta��o eta
eta0=0.5*eta;
eta_est=eta0;
ti_rr=10; %tempo de in�cio de varia��o da resist�ncia rot�rica
tf_rr=15; %tempo final de varia��o da resist�ncia rot�rica
rr_final=1.3*rr;
delta_rr=(rr_final-rr)/((tf_rr-ti_rr)/Tsc);

%resistencia estat�rica
rs_min=rs-0.3*rs; %valor de satura��o inferior
rs_max=rs+0.3*rs; %valor de satura��o superior


rs_est=0; % resistencia estat�rica estimada

Kp_rs=0; %ganho do controlador KP de adapta��o rs
Ki_rs=0.8;%ganho do controlador integral de adapta��o rs
inte_ra=0;
rs0=0.8*rs;% valor inicial da resist�ncia estat�rica estimada
rs_filt=rs0;
rs_final=1.2*rs; % supondo que a resistencia ir� subir 20%
delta_rs=(rs_final-rs)/((tf_rr-ti_rr)/Tsc); %incremento de rs para chegar a 1.2rs
w_filt=1e-3; %valor inicial do w_filtrado, n�o pode ser zero, sen�o d� divis�o por zero



%
thetac=0;
wc=0;
Va=0;
Vb=0;
Vc=0;
PI_iq=0;
PI_id=0;

Q_v=zeros(1,length(t));
Qs_v=zeros(1,length(t));

Pmod_v=zeros(1,length(t));
Pad_v=zeros(1,length(t));

ia_v=zeros(1,length(t));
ib_v=zeros(1,length(t));
ic_v=zeros(1,length(t));

for k=1:npsd %din�mica discreta do MIT
    
    %simulando a altera��o da resist�ncia rr  e rs
    if(k> ti_rr/Tsc && k<tf_rr/Tsc) %subindo a resist�ncia no intervalo de 10 at� 15s
        %rr=rr+delta_rr; %variando a resist�nica rot�rica
        %rs=rs+delta_rs; %variando a resist�nica estat�rica
    end
    eta=rr/Lr;
    Req=rs+rr*(LM/Lr)^2;
    gamma=Req/Leq;
    
    for kk=1:npc %din�mica cont�nua do MIT
        %convers�o das tens�es abc para qd0
        Vqs=2/3*(Va*cos(thetac)+Vb*cos(thetac-2*pi/3)+Vc*cos(thetac+2*pi/3));
        Vds=2/3*(Va*sin(thetac)+Vb*sin(thetac-2*pi/3)+Vc*sin(thetac+2*pi/3));
        
        %d(i_qs)/dt
        A(1,1)=-gamma;       %iqs
        A(1,2)=-wc;           %ids
        A(1,3)=delta*eta;    %iqn
        A(1,4)=-delta*wr;    %idn
        
        %d(i_ds)/dt
        A(2,1)=wc;            %iqs
        A(2,2)=-gamma;       %ids
        A(2,3)=delta*wr;     %iqn
        A(2,4)=delta*eta;    %idn
        
        %d(iqm)/dt
        A(3,1)=eta;          %iqs
        A(3,2)=0;            %ids
        A(3,3)=-eta;         %iqn
        A(3,4)=-(wc-wr);      %idn
        
        %d(idm)/dt
        A(4,1)=0;            %iqs
        A(4,2)=eta;          %ids
        A(4,3)=(wc-wr);       %iqn
        A(4,4)=-eta;         %idn
        
        
        U=[Vqs;Vds]; %entrada de tens�o no instante k*Tsc
        
        %Solu��o da Eq. Diferencial Utilizando Euler (Modelo El�trico)
        X=(I+A*dt)*Xi+dt*B*U;
        Xi=X;
        
        iqs=X(1);
        ids=X(2);
        iqn=X(3);
        idn=X(4);
        
        lambda_qr=LM*iqn;
        lambda_dr=LM*idn;
        
        iqr=lambda_qr/Lr-LM/Lr*iqs;
        idr=lambda_dr/Lr-LM/Lr*ids;
        lambda_qs=(Ls-LM^2/Lr)*iqs+LM/Lr*lambda_qr;
        Lambda_ds=(Ls-LM^2/Lr)*ids+LM/Lr*lambda_dr;
        
        iqm=iqs+iqr;
        idm=ids+idr;
        
        %Te=(3/2)*(P/2)*LM^2*rr/(Lr^2*(w-wr))*(iqs)^2;
        Te=(3/2)*(P/2)*LM*(iqs*idr-ids*iqr); %eq. geral de torque apenas em termos de correntes
        %Te(n)=(3/2)*P/2*LM^2*ids(n)*iqs(n)/Lr;
        Um=[Te;TL(k)]; %entrada de torque no instante n
        
        %Solu��o da Eq. Diferencial Utilizando Euler (Modelo Mec�nico)
        Xm=(Im+Am*dt)*Xmi+dt*Bm*Um;      
        Xmi=Xm;
        wr=Xm(1);          %leitura da velocidade rot�rica atrav�s do sensor


        %c�lculo da pot�ncia de entrada da rede
        %Pin(n)=3/2*sqrt((PI1)^2+(PI2)^2)*sqrt((iqs(n))^2+(ids(n))^2); %pot�ncia aparente
        Pin=3/2*(Vqs*iqs+Vds*ids); %pot�ncia ativa sugada da rede
        Energia = Energia +Pin*dt; %energia consumida da rede
        
        %alinhamento do sistema de refer�ncia do modelo de tempo cont�nuo
        if (lambda_dr ~= 0 )%pular os primeiros loops, onde lambda_dr � zero pois o fluxo lambda_dr vale zero e vai levar w para o infinito
            wc=wr+LM*rr*iqs/(Lr*lambda_dr); %velocidade do sistema de referencia para alinhar como o fluxo "d" do rotor (exato)
        end
        
        %wc=0;
        thetac=thetac+dt*wc;  %integra��o da velocidade para determinar a posi��o angular do sistema de ref
                
        %c�lculo das correntes abc do motor
        if(kk==floor(npc/2))
            ia=iqs*cos(thetac)+ids*sin(thetac);
            ib=iqs*cos(thetac-2*pi/3)+ids*sin(thetac-2*pi/3);
            ic=iqs*cos(thetac+2*pi/3)+ids*sin(thetac+2*pi/3);
        end
        
        %filtrar abc na frequencia de corte 300hz
    end
    
    %convertendo as correntes abc para qd com o alinhamento estimado do
    %controlador
    iqsd=2/3*(ia*cos(theta)+ib*cos(theta-2*pi/3)+ic*cos(theta+2*pi/3));
    idsd=2/3*(ia*sin(theta)+ib*sin(theta-2*pi/3)+ic*sin(theta+2*pi/3));
    
    %estimandor a corrente idm
    idm_est=idm_est +Tsc*eta_est*(idsd-idm_est);
    %idm_est=idm_est +Tsc*eta*(idsd-idm_est);
    
    %alinhamento ao fluxo rot�rico
    if (idm_est ~= 0 )%evitar divis�o por zero
        w=wr+eta_est*iqsd/idm_est; %velocidade do sistema de referencia para alinhar como o fluxo "d" do rotor 
        %w=wr+eta*iqsd/idm_est; %velocidade do sistema de referencia para alinhar como o fluxo "d" do rotor (exato)
    end   
    
    theta=theta+Tsc*w;  %integra��o da velocidade para determinar a posi��o angular do sistema de ref
    
   %c�culo do controladores PI
    %controle de corrente ids
    erro_id=ids_ref-idsd; 
    inte_id = inte_id+erro_id*Tsc;
    PI_id=Kp_id*erro_id+Ki_id*inte_id;
    
    %controle de velocidade
    erro_w=w_ref_v(k)-wr;
    inte_w = inte_w+erro_w*Tsc;
    PI_w=Kp_w*erro_w+Ki_w*inte_w;
    iqs_ref=PI_w;

    %controle de corrente iqs
    erro_iq=iqs_ref-iqsd;
    inte_iq = inte_iq+erro_iq*Tsc;
    PI_iq=Kp_iq*erro_iq+Ki_iq*inte_iq;
    %end
    
%  
%     %implementa��o de otimiza��o pelo m�todo de rosenbrock
%     if ( t(k)> t_inicio_rosembrock)
%         if(abs(erro4)<3.0) %implementa��o do controlador de efici�ncia pelo m�todo de Rosembrock
%             rosembrock(k)=1;
%             if(estabiliza>t_estabiliza)
%                 estabiliza=0;
%                 P1=P2;
%                 P2=Pin;
%                 DeltaP=P2-P1;
%                 if(DeltaP>0)
%                     kr=-kr/2;
%                 end      
%                 lambda_ref=lambda_ref-kr*delta_lambda;            
%             else
%                 estabiliza=estabiliza+dt*npc;
%             end
%         else
%             rosembrock(k)=0;
%             estabiliza=0;
%             lambda_ref=lambda_nominal;
%             kr=1;
%         end
%     end

    %implementa��o de otimiza��o controlador de efici�ncia LMC
    if ( t(k)> t_inicio_otimo)
        %ids_ref=0.99*ids_ref+0.01*iqsd*sqrt(1+rr*LM^2/(rs*Lr^2)); % (real)
        ids_ref=0.99*ids_ref+0.01*iqsd*sqrt(1+eta_est*LM^2/(rs_filt*Lr)); % (estimado)
        %ids_ref=0.99*ids_ref+0.01*iqsd*sqrt(1+eta_est*LM^2/(rs*Lr)); % (estimado)
        %ids_ref=0.99*ids_ref+0.01*iqsd*sqrt(1+eta*LM^2/(rs*Lr)); % (estimado)
    end 
   
    Vqsd=PI_iq;
    Vdsd=PI_id;
    
    %estimador de eta    
    Q=3/2*(Vqsd*idsd-Vdsd*iqsd); %modelo de refer�ncia
    Qs=3/2*w*sigma*Ls*(idsd^2+iqsd^2+delta*idsd*idm_est); %Modelo
    %Qs=0.9713*3/2*w*sigma*Ls*(idsd^2+iqsd^2+delta*idsd*idm_est); %Modelo
    %Qs=3/2*w*sigma*Ls*(idsd^2+iqsd^2+delta*idsd*idm_est); %Modelo
    %adaptativo% obs!, o eta encontra-se em w em que 0.97697 � a rela��o
    %Q/Qs em regime (me parece que se deve ao erro de alinhamento com o fluxo rot�rico)
    %Qs=3/2*(wr+eta_est*iqsd/idm_est)*sigma*Ls*(idsd^2+iqsd^2+delta*idsd*idm_est);
    
    Q_v(k)=Q;
    Qs_v(k)=Qs;
    
    DeltaQ=Qs-Q;
    if abs(DeltaQ)>K_eta*300 DeltaQ=sign(DeltaQ)*K_eta*300; end
        
    integraQ=integraQ-K_eta*DeltaQ*Tsc;
    eta_est=eta0+integraQ;
    
    eta_est_v(k)=eta_est;
    eta_ref_v(k)=eta;

     %estimador de rs
     %modelo de refer�ncia
     Pmod=3/2*(Vqsd*iqsd+Vdsd*idsd);
     
     %modelo adaptativo
     %Pad=1.046*3/2*(rs_est*iqsd^2+rs_est*idsd^2+w*LM^2*iqsd*idsd/Lr); %1.115 corre��o do modelo adaptativo
     %Pad=1.0456*3/2*(rs_est*iqsd^2+rs_est*idsd^2+w*LM^2*iqsd*idsd/Lr); %1.115 corre��o do modelo adaptativo
     Pad=3/2*(rs_est*iqsd^2+rs_est*idsd^2+w*LM^2*iqsd*idsd/Lr); %1.115 corre��o do modelo adaptativo
     %Pad=3/2*(rs*iqsd^2+rs*idsd^2+w*LM^2*iqsd*idsd/Lr);
    
     Pmod_v(k)=Pmod;
     Pad_v(k)=Pad;
     
     erro_P=Pmod-Pad;
     if abs(erro_P)>0.2 erro_P=0.2*sign(erro_P); end

     inte_ra = inte_ra+erro_P*Tsc;
     rs_est=Kp_rs*erro_P+Ki_rs*inte_ra+rs0;
     if rs_est > rs_max %saturando a estima��o do rs superior
         rs_est=rs_max;
         inte_ra = inte_ra-erro_P*Tsc; %n�o acumula integrador
     end
     if rs_est < rs_min %saturando a estima��o do rs inferior
         rs_est=rs_min;
         inte_ra = inte_ra-erro_P*Tsc;%n�o acumula integrador
     end
     
     rs_filt=0.999*rs_filt+0.001*rs_est;
     
    %convertendo as tens�es qd para abc -> papel do inversor
    Va=Vqsd*cos(theta)+Vdsd*sin(theta);
    Vb=Vqsd*cos(theta-2*pi/3)+Vdsd*sin(theta-2*pi/3);
    Vc=Vqsd*cos(theta+2*pi/3)+Vdsd*sin(theta+2*pi/3);
     
    %armazenando vari�veis para plotar
    iqs_v(k)=iqsd;
    ids_v(k)=idsd;
    ids_ref_v(k)=ids_ref;
    Te_v(k)=Te;
    wr_v(k)=wr;
    lambda_dr_v(k)=lambda_dr;
    lambda_dr_est_v(k)=lambda_dr_est;
    lambda_ref_v(k)=ids_ref*LM;
    theta_v(k)=theta;
    Vqs_v(k)=Vqsd;
    Vds_v(k)=Vdsd;
    Pin_v(k)=Pin;
    idm_est_v(k)=idm_est;
    rs_v(k)=rs;
    rs_est_v(k)=rs_est;
    rs_filt_v(k)=rs_filt;
    ia_v(k)=ia;
    ib_v(k)=ib;
    ic_v(k)=ic;
end
%%
figure;
plot(t,ia_v,t,ib_v,t,ic_v);
title('Correntes abc real');
legend('i_a','i_b','i_c');
grid;

figure;
plot(t,Q_v,t,Qs_v);
title('Potencias reativas');
legend('Q','Qs');
grid;

figure;
plot(t,Pmod_v,t,Pad_v,t,Pin_v);
title('Potencias ativas');
legend('P_{modelo}','P_{adaptativo}','P_{real}');
grid;


figure;
plot(t,Vqs_v,t,Vds_v);
legend('Vq','Vd');
title('Tens�es no Sistema qd0'); 
grid;


figure;
wr_rpm=(2/P)*wr_v*60/(2*pi);
plot(t,wr_rpm, t,w_ref_fisico_v)
title('Rota��o'); grid on;
xlabel('tempo'); ylabel('rota��o rpm');


figure;
plot(t,iqs_v, t,ids_v);
title('Correntes do Estator e do Rotor (qd0)'); grid on;
legend('i_{qs}','i_{ds}');

figure;
plot(t,idm_v, t,idm_est_v);
title('Correntes idm '); grid on;
legend('i_{dm}','i_{dm-est}');

% %recuperando o sistema original de corrente a partir do sistema qd0
% %observe que a sequencia � q-d-0
% 
% Iabcs=zeros(3,length(t));
% Vabc=zeros(3,length(t));
%  
% for k = 1:length(t)
% Ksi=[cos(theta_v(k)) sin(theta_v(k)) 1;
%      cos(theta_v(k)-2*pi/3) sin(theta_v(k)-2*pi/3) 1;
%      cos(theta_v(k)+2*pi/3) sin(theta_v(k)+2*pi/3) 1];  
% Iabcs(:,k)= Ksi*[iqs_v(k);ids_v(k);0];
% Vabc(:,k)= Ksi*[Vqs_v(k);Vds_v(k);0];
% 
% end
% 
% figure;
% plot(t,Iabcs,t,ia_v,t,ib_v,t,ic_v);
% legend('Ia','Ib','Ic','Iar','Ibr','Icr');
% title('Correntes no sistema abc'); 
% grid;
% 
% 
% figure;
% plot(t,Vabc);
% legend('Va','Vb','Vc');
% title('Tens�es no sistema abc'); 
% grid;


figure;
plot(t,Te_v);
title('Torque Eletromagnetico');
legend('Te');
xlabel('Tempo (s)'); ylabel('Te'); grid on;


% figure;
% plot(wr_rpm,Te)
% title('Curva Torque x Rota��o');
% grid on;
% xlabel('Rota��o rpm'); ylabel('Torque (Nm)');


figure;
plot(t,ids_v,t,ids_ref_v)
legend('ids','ids_{ref}');
title('Corrente i_ds (respons�vel pela magnetiza��o)');
xlabel('tempo (s)'); ylabel('corrente (A)'); grid on;


% figure;
% plot(t,Pin_v)
% legend('P_{in}');
% title('Pot�ncia de Entrada');
% xlabel('tempo (s)'); ylabel('Pot�ncia'); grid on;
% 
% P_eletrica=Te.*wr*(2/P);
% figure;
% plot(t,Pin,t,P_eletrica)
% legend('P_{entrada rede}','P_{eletrica eixo}');
% title('Pot�ncia de Entrada');
% xlabel('tempo (s)'); ylabel('Pot�ncia'); grid on;

figure;
plot(t,eta_est_v,t,eta_ref_v)
title('Estima��o do \eta');
legend('\eta_{est}', '\eta_{ref}');
xlabel('tempo (s)'); ylabel('\eta'); grid on;

figure;
plot(t,rs_est_v,t,rs_v,t,rs_filt_v);
title('Estima��o do rs');
legend('rs_{est}', 'rs', 'rs_{filt-est}');
xlabel('tempo (s)'); ylabel('rs'); grid on;


% Energia
% 
% Pin(length(t))
% lambda_ref(length(t))
% iqs(length(t))
% ids(length(t))

% % O c�digo a seguir salva as vari�veis de interese para compara��o futura.
% MF_wr_rpm_SemPFe=wr_rpm;
% MF_iqs_SemPFe=iqs_v;
% MF_ids_SemPFe=ids_v;
% MF_Te_SemPFe=Te_v;
% MF_lambda_dr_SemPFe=lambda_dr_v;
% MF_lambda_dr_est_SemPFe=lambda_dr_est_v;
% MF_Pin_SemPFe=Pin_v;
% 
% save('MF_SemPFe','t','MF_wr_rpm_SemPFe','MF_iqs_SemPFe','MF_ids_SemPFe','MF_Te_SemPFe','MF_lambda_dr_SemPFe','MF_lambda_dr_est_SemPFe','MF_Pin_SemPFe');

