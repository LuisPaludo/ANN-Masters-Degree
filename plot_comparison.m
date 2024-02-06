% Carregar primeiro workspace (IFOC)
load('IFOC.mat');
% Armazenar variáveis de interesse
t_IFOC = t;
Rs_vetor_IFOC = Rs_vetor;
Rr_vetor_IFOC = Rr_vetor;
ids_vetor_IFOC = ids_vetor;
Ids_ref_IFOC = Ids_ref;
Te_vetor_IFOC = Te_vetor;
Tl_IFOC = Tl;
lambda_dr_est_vetor_IFOC = lambda_dr_est_vetor;
lambda_dr_vetor_IFOC = lambda_dr_vetor;
wr_vetor_IFOC = wr_vetor;
w_ref_IFOC = w_ref;

% Limpar variáveis para evitar conflitos
clearvars -except t_IFOC Rs_vetor_IFOC Rr_vetor_IFOC ids_vetor_IFOC Ids_ref_IFOC Te_vetor_IFOC Tl_IFOC lambda_dr_est_vetor_IFOC lambda_dr_vetor_IFOC wr_vetor_IFOC w_ref_IFOC

% Carregar segundo workspace (IFOC_ANN)
load('IFOC_ANN.mat');

figure
% Plotando os dados
plot(t,ids_vetor, 'LineWidth', 2, 'DisplayName', 'Estimado', 'Color', [0.3 0.3 0.3]); % tom de cinza escuro
hold on;
plot(t,ids_vetor_IFOC, 'LineWidth', 2, 'DisplayName', 'Real', 'Color', [0.7 0.7 0.7]); % tom de cinza claro

