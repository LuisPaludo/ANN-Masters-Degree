% close all;
% clear all;
% close all;
clc;

%% Parâmetros da Simulação

f = 10000;                                     % Frequencia de amostragem do sinal
Tsc = 1/f;                                     % Periodo de amostragem do sinal
p = 10;                                        % Numero de partes que o intervalo discreto e dividido
h = Tsc/p;                                     % Passo de amostragem continuo
Tsimu = 30;                                    % Tempo de Simulação
Np = Tsimu/Tsc;                                % Número de Pontos (vetores)

%% Parâmetros do Motor

Polos = 8;                                   % Numero de polos
Vf = 127;
frequencia = 60;                             % Frequência elétrica nominal
Rs = 0.6759;                                 % Resistência do estator
Rs_nom = 0.6759;
Lls = 0.00280;                               % Indutância de dispersão dos enrolamentos do estator
Lm = 0.0387;                                 % Indutância mútua dos enrolamentos - estator/estator - estator/rotor - rotor/rotor
Llr = 0.00280;                               % Indutância de dispersão dos enrolamentos do rotor
Lr = Lm + Llr;                               % Indutância dos enrolamentos do rotor
Ls = Lm + Lls;                               % Indutância dos enrolamentos do estator
Rr = 0.2615;                                 % Resistencia do rotor
Rr_nom = 0.2615;
J =  2*0.08876;                              % Inércia do motor
K = 0.12;                                    % Coeficiente de atrito do eixo do motor
weles = 2*pi*frequencia;                     % Velocidade sincrona em radianos elétricos por segundo
wr = 0;                                      % Velocidade inicial
P = 4*736;                                   % Potência do motor

%% Reatâncias para espaço de estados

Xm = weles*Lm;                               % Reatância de magnetização
Xls = weles*Lls;                             % Reatância de dispersão do estator
Xlr = weles*Llr;                             % Reatância de dispersão do rotor
Xml = 1/(1/Xm + 1/Xls + 1/Xlr);

%% Constantes para solução mecânica do motor

A =  - K/J;
B1 =   Polos/(2*J);
B2 = - Polos/(2*J);

%% Ganhos Controladores

KP_w = 0.5949;
KI_w = 2.9474;

KP_id = 13;
KI_id = 662;

KP_iq = 13;
KI_iq = 662;

%% Inicialização das variaveis

Fqs = 0;
Fds = 0;
Fqr = 0;
Fdr = 0;
Ids = 0;
Ids_ant = 0;
Iqs = 0;
Vq = 0;
Vd = 0;
lambda_dr_est = 0;
theta = 0;
UI_w = 0;
UI_id = 0;
UI_iq = 0;
Te = 0;
ialpha_vetor = zeros(1,Np);
ibeta_vetor = zeros(1,Np);
iqs_vetor = zeros(1,Np);
ids_vetor = zeros(1,Np);
Ids_ref_vetor = zeros(1,Np);
Te_vetor = zeros(1,Np);
wrpm_vetor = zeros(1,Np);
wr_vetor = zeros(1,Np);
w_ref_vetor = zeros(1,Np);
t = 0:Tsc:Np*Tsc-Tsc;
Va_vetor = zeros(1,Np);
Vb_vetor = zeros(1,Np);
Vc_vetor = zeros(1,Np);
Va_prediction_vetor = zeros(1,Np);
Vb_prediction_vetor = zeros(1,Np);
Vc_prediction_vetor = zeros(1,Np);
Vd_vetor = zeros(1,Np);
Vd_prediction_vetor = zeros(1,Np);
Vq_vetor = zeros(1,Np);
Vq_prediction_vetor = zeros(1,Np);
lambda_qr_vetor = zeros(1,Np);
lambda_dr_vetor = zeros(1,Np);
lambda_dr_est_vetor = zeros(1,Np);
Ia_vetor = zeros(1,Np);
Ib_vetor = zeros(1,Np);
Ic_vetor = zeros(1,Np);
U_w_vetor = zeros(1,Np);
U_iq_vetor = zeros(1,Np);
U_id_vetor = zeros(1,Np);
e_w_vetor = zeros(1,Np);
e_iq_vetor = zeros(1,Np);
e_id_vetor = zeros(1,Np);
Rs_vetor = zeros(1,Np);
Rr_vetor = zeros(1,Np);

%% Torque de Carga

Tn = P*Polos/(2*weles);

% Tl = Tn*((t-30).*heaviside(t-30)-(t-31).*heaviside(t-31)) - 0.5*Tn*((t-32).*heaviside(t-32)-(t-33).*heaviside(t-33)) ...
%     - Tn*((t-34).*heaviside(t-34)-(t-35).*heaviside(t-35)) + Tn/15*((t-35).*heaviside(t-35)-(t-50).*heaviside(t-50));

Tl = 2*Tn*((t-2).*heaviside(t-2)-(t-2.5).*heaviside(t-2.5));

%% Corrente Id de referência

lambda_nonminal = 127/(2*pi*frequencia)/(Lm);

Ids_ref = lambda_nonminal*((t-0).*heaviside(t-0)-(t-1).*heaviside(t-1));

% Ids_ref = 5*((t-0).*heaviside(t-0)-(t-1).*heaviside(t-1)) + 2*10*((t-2).*heaviside(t-2)-(t-2.1).*heaviside(t-2.1))  ...
%     + 10/2*((t-3).*heaviside(t-3)-(t-5).*heaviside(t-5)) - 15/2*((t-5).*heaviside(t-5)-(t-7).*heaviside(t-7)) ...
%     - 10/2*((t-7).*heaviside(t-7)-(t-9).*heaviside(t-9)) + 8*((t-9).*heaviside(t-9)-(t-10).*heaviside(t-10)) ...
%     + lambda_nonminal*((t-10).*heaviside(t-10)-(t-11).*heaviside(t-11)) - lambda_nonminal/10*((t-17).*heaviside(t-17)-(t-27).*heaviside(t-27)) ...
%     + lambda_nonminal*((t-27).*heaviside(t-27)-(t-28).*heaviside(t-28)) - lambda_nonminal/5*((t-35).*heaviside(t-35)-(t-40).*heaviside(t-40)) ...
%     + lambda_nonminal/5*((t-40).*heaviside(t-40)-(t-45).*heaviside(t-45));

%% Velocidade de Referência

w_nom_ref = 2*pi*60;

w_ref = w_nom_ref*10*((t-3).*heaviside(t-3)-(t-3.1).*heaviside(t-3.1)) - w_nom_ref*((t-4).*heaviside(t-4)-(t-5).*heaviside(t-5)) ...
    - w_nom_ref*((t-5.3).*heaviside(t-5.3)-(t-5.5).*heaviside(t-5.5)) + w_nom_ref*((t-15).*heaviside(t-15)-(t-16.2).*heaviside(t-16.2));

% w_ref = w_nom_ref*((t-12).*heaviside(t-12)-(t-13).*heaviside(t-13)) - 0.9*w_nom_ref*((t-13).*heaviside(t-13)-(t-14).*heaviside(t-14)) ...
%     - 0.1*w_nom_ref*((t-14).*heaviside(t-14)-(t-15).*heaviside(t-15)) + w_nom_ref*10*((t-15).*heaviside(t-15)-(t-15.1).*heaviside(t-15.1)) ...
%     - 2*w_nom_ref*10*((t-15.3).*heaviside(t-15.3)-(t-15.4).*heaviside(t-15.4)) + w_nom_ref*10*((t-16).*heaviside(t-16)-(t-16.2).*heaviside(t-16.2)) ...
%     - 2*w_nom_ref/5*((t-35).*heaviside(t-35)-(t-40).*heaviside(t-40)) + 2*w_nom_ref/5*((t-40).*heaviside(t-40)-(t-45).*heaviside(t-45));

%% Importar Modelo ONNX

modelFolder = './saved_model/my_model_7';

scalerData = load('./saved_model/data/my_model_7/scaler_values.mat');

net = importNetworkFromTensorFlow(modelFolder);

%% Loop Simulação Motor
for k = 1:Np

    if(k/Np == 0.25)
        disp('25%')
    end

    if(k/Np == 0.5)
        disp('50%')
    end

    if(k/Np == 0.75)
        disp('75%')
    end

    if(k/Np == 1)
        disp('100%')
    end

    %% Estimador de fluxo rotórico e orientação do sist. de referência

    lambda_dr_est = lambda_dr_est*((2*Lr-Tsc*Rr)/(2*Lr+Tsc*Rr)) + Ids*((Lm*Rr*Tsc)/(2*Lr+Rr*Tsc)) + Ids_ant*((Lm*Rr*Tsc)/(2*Lr+Rr*Tsc));
    Ids_ant = Ids;

    if(lambda_dr_est > 0.1)
        wsl = (Lm*Rr*Iqs)/(Lr*lambda_dr_est);
    else
        wsl = 0;
    end

    wr_est = wr + wsl;
    w = wr_est;
    theta = theta + Tsc*w;

    %% Calculando as correntes Ia Ib e Ic

    Ialpha = Iqs*cos(theta) + Ids*sin(theta);
    Ibeta = -Iqs*sin(theta) + Ids*cos(theta);

    % % Transf. inversa clarke
    Ia = Ialpha;
    Ib = -0.5*Ialpha - sqrt(3)/2*Ibeta;
    Ic = -0.5*Ialpha + sqrt(3)/2*Ibeta;

    %Velocidade
    e_w = w_ref(k) - wr;
    UI_w = UI_w + e_w*Tsc;
    U_w = KP_w*e_w + KI_w*UI_w;
    iqs_ref = U_w;

    % %Servos de corrente
    e_id = Ids_ref(k) - Ids;
    UI_id = UI_id + e_id*Tsc;
    U_Id = KP_id*e_id + KI_id*UI_id;

    if(U_Id >= 127*sqrt(2))
        U_Id = 127*sqrt(2);
    end
    if(U_Id <= -127*sqrt(2))
        U_Id = -127*sqrt(2);
    end

    e_iq = iqs_ref - Iqs;
    UI_iq = UI_iq*e_iq*Tsc;
    U_Iq = KP_iq*e_iq + KI_iq*UI_iq;

    if(U_Iq >= 127*sqrt(2))
        U_Iq = 127*sqrt(2);
    end
    if(U_Iq <= -127*sqrt(2))
        U_Iq = -127*sqrt(2);
    end

    % Vq = U_Iq;
    % Vd = U_Id;

    % Predição

    % Seus dados de entrada
    input_data = [e_iq, e_id, Iqs, Ids]; % e outros valores de entrada

    % Normalizar os dados de entrada
    normalized_data = normalizeInput(input_data, scalerData.x_min, scalerData.x_max);

    % Fazer a previsão usando a rede neural com dados normalizados
    predictions_normalized = predict(net, normalized_data);

    % Desnormalizar os dados de saída
    Vq_prediction = denormalizeOutput(predictions_normalized(1), scalerData.y_min(1), scalerData.y_max(1));
    Vd_prediction = denormalizeOutput(predictions_normalized(2), scalerData.y_min(2), scalerData.y_max(2));

    Vq = Vq_prediction;
    Vd = Vd_prediction;

     %% Calculando as Tensões Va Vb e Vc

    Valfa = Vq*cos(theta) + Vd*sin(theta);
    Vbeta = -Vq*sin(theta) + Vd*cos(theta);

    %Transf. inversa clarke
    Va = Valfa;
    Vb = -0.5*Valfa - sqrt(3)/2*Vbeta;
    Vc = -0.5*Valfa + sqrt(3)/2*Vbeta;

    %% Calculando as Tensões Va Vb e Vc (Predição)

    Valfa_prediction = Vq_prediction*cos(theta) + Vd_prediction*sin(theta);
    Vbeta_prediction = -Vq_prediction*sin(theta) + Vd_prediction*cos(theta);

    %Transf. inversa clarke
    Va_prediction = Valfa_prediction;
    Vb_prediction = -0.5*Valfa_prediction - sqrt(3)/2*Vbeta_prediction;
    Vc_prediction = -0.5*Valfa_prediction + sqrt(3)/2*Vbeta_prediction;

    Vmax = 127*sqrt(2);

    if abs(Va) > Vmax || abs(Vb) > Vmax || abs(Vc) > Vmax
        % Calcule o fator de redução baseado na maior tensão
        scalingFactor = Vmax / max(abs([Va, Vb, Vc]));

        Vd = Vd * scalingFactor;
        Vq = Vq * scalingFactor;

        % Recalcule Va, Vb e Vc com os novos Vd e Vq se desejar
        Valfa = Vq*cos(theta) + Vd*sin(theta);
        Vbeta = -Vq*sin(theta) + Vd*cos(theta);

        Va = Valfa;
        Vb = -0.5*Valfa - sqrt(3)/2*Vbeta;
        Vc = -0.5*Valfa + sqrt(3)/2*Vbeta;
    end

    Rs = Rs + Rs_nom*0.5/Np;
    Rr = Rr + Rr_nom*0.5/Np;

    for ksuper=1:p

        Fqm = Xml/Xls*Fqs + Xml/Xlr*Fqr;
        Fdm = Xml/Xls*Fds + Xml/Xlr*Fdr;

        Fqs = Fqs + h*weles*(Vq - w/weles*Fds - Rs/Xls*(Fqs-Fqm));
        Fds = Fds + h*weles*(Vd + w/weles*Fqs - Rs/Xls*(Fds-Fdm));

        Fqr = Fqr - h*weles*((w-wr)*Fdr/weles + Rr/Xlr*(Fqr-Fqm));
        Fdr = Fdr - h*weles*((wr-w)*Fqr/weles + Rr/Xlr*(Fdr-Fdm));

        Iqs = (Fqs-Fqm)/Xls;
        Ids = (Fds-Fdm)/Xls;

        Te = 3/2*Polos/2*1/weles*(Fds*Iqs - Fqs*Ids);

        % Solução mecânica

        wr = wr + h*(A*wr + B1*Te + B2*Tl(k));
        wrpm = wr*2/Polos*60/(2*pi);

    end

    iqs_vetor(k) = Iqs;
    ids_vetor(k) = Ids;
    ialpha_vetor(k) = Ialpha;
    ibeta_vetor(k) = Ibeta;
    Te_vetor(k) = Te;
    wrpm_vetor(k) = wrpm;
    wr_vetor(k) = wr;
    Va_vetor(k) = Va;
    Vb_vetor(k) = Vb;
    Vc_vetor(k) = Vc;
    Va_prediction_vetor(k) = Va_prediction;
    Vb_prediction_vetor(k) = Vb_prediction;
    Vc_prediction_vetor(k) = Vc_prediction;
    Ia_vetor(k) = Ia;
    Ib_vetor(k) = Ib;
    Ic_vetor(k) = Ic;
    Vd_vetor(k) = Vd;
    Vd_prediction_vetor(k) = Vd_prediction;
    Vq_vetor(k) = Vq;
    Vq_prediction_vetor(k) = Vq_prediction;
    lambda_qr_vetor(k) = Fqr/weles;
    lambda_dr_vetor(k) = Fdr/weles;
    lambda_dr_est_vetor(k) = lambda_dr_est;
    U_w_vetor(k) = U_w;
    U_iq_vetor(k) = UI_iq;
    U_id_vetor(k) = UI_id;
    e_w_vetor(k) = e_w;
    e_iq_vetor(k) = e_iq;
    e_id_vetor(k) = e_id;
    Rs_vetor(k) = Rs;
    Rr_vetor(k) = Rr;

end
% dados_treinamento = [transpose(iqs_vetor), transpose(ids_vetor), transpose(Ids_ref), transpose(Te_vetor), transpose(Tl), transpose(wr_vetor), transpose(w_ref), ...
%     transpose(Vq_vetor), transpose(Vd_vetor), transpose(e_w_vetor), transpose(e_iq_vetor), transpose(e_id_vetor), ...
%     transpose(U_w_vetor), transpose(U_iq_vetor), transpose(U_id_vetor)];
% 
% % dados_treinamento = [transpose(e_iq_vetor), transpose(e_id_vetor), transpose(U_iq_vetor), transpose(U_id_vetor)];
% 
% headers = {'iqs', 'ids', 'Ids_ref', 'Te','TL', 'wr', 'w_ref', 'Vq', 'Vd', 'e_w', 'e_iq', 'e_id', 'U_w', 'U_iq', 'U_id'};
% 
% % headers = {'e_iq', 'e_id', 'U_iq', 'U_id'};
% 
% T = array2table(dados_treinamento, 'VariableNames', headers);
% 
% writetable(T, 'data/dados_validacao_1.csv');

% Configurações gerais para os gráficos
lineWidth = 2;
fontSize = 14;
fontAxis = 12;


% % Gráfico 2: wr_vetor e w_ref
% figure(1);
% plot(t, wr_vetor, t, w_ref, 'LineWidth', lineWidth);
% title('Wr vs Ref', 'FontSize', fontSize);
% xlabel('Tempo (s)', 'FontSize', fontAxis);
% ylabel('Valores', 'FontSize', fontAxis);
% legend('Wr', 'Ref');

% Gráfico 2: wr_vetor e w_ref
figure;
plot(t, wr_vetor, t, w_ref, 'LineWidth', lineWidth);
title('Wr vs Ref', 'FontSize', fontSize);
xlabel('Tempo (s)', 'FontSize', fontAxis);
ylabel('Valores', 'FontSize', fontAxis);
legend('Wr', 'Ref');

% % Gráfico 3: Tensões Va, Vb e Vc
% figure;
% plot(t, Va_vetor,t, Vb_vetor,t, Vc_vetor, 'LineWidth', lineWidth);
% title('Tensões Va, Vb e Vc', 'FontSize', fontSize);
% xlabel('Tempo (s)', 'FontSize', fontAxis);
% ylabel('Tensão (V)', 'FontSize', fontAxis);
% legend('Va', 'Vb', 'Vc');

% figure
% % Plotando os dados
% plot(t,Va_prediction_vetor,t,Vb_prediction_vetor,t,Vc_prediction_vetor); % tom de cinza escuro
% legend('Va_p','Vb_p','Vc_p')

%
% figure
% % Plotando os dados
% plot(t,Vd_vetor,t,Vq_vetor,t,Vd_prediction_vetor,t,Vq_prediction_vetor,'LineWidth',2); % tom de cinza escuro
% legend('Vd','Vq','Vd Prediction','Vq Prediction')
% 
% figure
% % Plotando os dados
% plot(t,Vd_vetor,t,Vq_vetor); % tom de cinza escuro
% legend('Vd','Vq')
%
figure
% Plotando os dados
plot(t,lambda_qr_vetor,t,lambda_dr_vetor,t,lambda_dr_est_vetor); % tom de cinza escuro
legend('qr','dr', 'dr - Est')

% figure
% % Plotando os dados
% plot(t,Ia_vetor,t,Ib_vetor,t,Ic_vetor); % tom de cinza escuro
% legend('Ia','Ib','Ic')
%
% figure
% % Plotando os dados
% plot(t,U_w_vetor); % tom de cinza escuro
% legend('U_w')
%
% figure
% % Plotando os dados
% plot(t,U_iq_vetor); % tom de cinza escuro
% legend('U iq')
%
% figure
% % Plotando os dados
% plot(t,U_id_vetor); % tom de cinza escuro
% legend('U id')
%
% figure
% % Plotando os dados
% plot(t,ids_vetor,t,iqs_vetor); % tom de cinza escuro
% legend('Ids','Iqs')
% 
% figure
% % Plotando os dados
% plot(t,iqs_vetor,t,U_w_vetor); % tom de cinza escuro
% legend('Iqs','Iqs Ref')
% 
% figure
% % Plotando os dados
% plot(t,iqs_vetor,t,Te_vetor); % tom de cinza escuro
% legend('Iqs','Te')

figure
plot(t,ids_vetor,t,Ids_ref,'LineWidth',2)
legend('Ids','Ref');

figure
plot(t,Te_vetor,t,Tl,'LineWidth',2)
legend('Te','Ref');

figure
plot(t,Rs_vetor,t,Rr_vetor,'LineWidth',2)
legend('Rs','Rr');


% figure
% % Plotando os dados
% plot(t,e_w_vetor); % tom de cinza escuro
% legend('e_w')
% 
% figure
% % Plotando os dados
% plot(t,e_id_vetor); % tom de cinza escuro
% legend('e_d')
% 
% figure
% % Plotando os dados
% plot(t,e_iq_vetor); % tom de cinza escuro
% legend('e_q')

save('IFOC_ANN','t','Rs_vetor','Rr_vetor','ids_vetor','Ids_ref','Te_vetor','Tl','lambda_dr_est_vetor','lambda_dr_vetor','wr_vetor','w_ref');


function normalized_data = normalizeInput(data, data_min, data_max)
normalized_data = (data - data_min) ./ (data_max - data_min);
end

function original_data = denormalizeOutput(normalized_data, data_min, data_max)
original_data = normalized_data .* (data_max - data_min) + data_min;
end
