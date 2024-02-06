data_training = readtable("data\dados_treinamento_4.csv");

X = [data_training.iqs, data_training.ids, data_training.Vq, data_training.Vd]';
Y = [data_training.wr]';

hiddenLayerSize = 64;
net = feedforwardnet(hiddenLayerSize,'trainlm');
net = train(net,X,Y);
y = net(X); 
save('saved_model\matlab\net')