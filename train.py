# def reset_weights(model):
#     for layer in model.modules(): # 수정함 module로 해야 sequential에서도 weight가 초기화가 됩니다
#         if hasattr(layer,'reset_parameters'):
#             layer.reset_parameters()
def pretrain_reset_weight(model): #pretrained model을 학습시킬 때, 
    for layer in model_resnet.modules():
        if isinstance(layer,nn.Linear):#fc부분만 reset한다. 
            layer.reset_parameters()
    #         print(layer)
#train function으로 생성 
def train(model, n_epochs, set_optimizer,optimizer, scheduler,early_stop_mode,criterion, train_loader,valid_loader,path=''):
    model.apply(pretrain_reset_weight)#학습할때 마다 weight를 reset시킵니다. 
#     model.load_state_dict(torch.load(f'pt/resnet/50/SGD_Momentum.pt'))
    model.to(device) #train_to GPU
    early_stop = 0
    result = {"train_loss":torch.zeros(n_epochs), "valid_loss":torch.zeros(n_epochs),
              "train_acc":torch.zeros(n_epochs),"valid_acc":torch.zeros(n_epochs),
              "lr_list":torch.zeros(n_epochs)}
    train_loss = torch.zeros(n_epochs)
    valid_loss = torch.zeros(n_epochs)
    train_acc  = torch.zeros(n_epochs)
    valid_acc  = torch.zeros(n_epochs)
    lr_list    = torch.zeros(n_epochs)
    train_loss_min = np.inf
    valid_loss_min = np.inf

    for e in trange(0, n_epochs):
        torch.cuda.empty_cache()#cuda memory를 clear
        early_stop+=1
        model.train()
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device) # cuda로 보냄 
            optimizer.zero_grad() #gradient계산 
            logits = model(data)#predict
            loss   = criterion(logits, labels)#loss구함
            loss.backward()#미분한다. 
            optimizer.step()#업데이트 
            result['train_loss'][e] += loss.item()#loss를 train에 삽입 

            ps = F.softmax(logits, dim=1)#확률을 구하고, 
            top_p, top_class = ps.topk(1, dim=1)#좋은 index를 뽑아낸다. 
            equals = top_class == labels.reshape(top_class.shape)#같은지를 비교하고, 
            result['train_acc'][e] += torch.mean(equals.type(torch.float)).detach().cpu()#정확도를 계산합니다. 
        #########################
        # finish train, caculate loss, acc#
        #########################
        result['train_loss'][e] /= len(train_loader)
        result['train_acc'][e] /= len(train_loader)
        ######################    
        # validate the model #
        ######################
        with torch.no_grad():#validation은 미분이 필요 없습니다. with를 사용하면 변수를 쓰고 지움 
            model.eval()#dropout을 끈다음 validation에 대해서 학습을 진행 
            for data, labels in valid_loader:
                data, labels = data.to(device), labels.to(device) #to cuda
                logits = model(data) #predict calculate
                loss = criterion(logits, labels) # loss 구하고, 
                result['valid_loss'][e] += loss.item()#valid_loss calculate
                ps = F.softmax(logits, dim=1)# ps calculate 
                top_p, top_class = ps.topk(1, dim=1) # index뽑고 
                equals = top_class == labels.reshape(top_class.shape) #정확도 계산 
                result['valid_acc'][e] += torch.mean(equals.type(torch.float)).detach().cpu() # cpu에서 연산합니다 .

        if set_optimizer == 0: #optimizer를 번갈아서 사용하기 위해서 만든 dumy 변수입니다. 
            scheduler.step()#cosine은 loss를 기준으로 하지 않아서 parameter존재하지 않음 
        elif set_optimizer == 1 or set_optimizer==2: #loss를 기준으로 scheduler update 
            scheduler.step(valid_loss[e])
        valid_loss[e] /= len(valid_loader)#loss
        valid_acc[e]  /= len(valid_loader)#정확도 
        lr_list[e]     =   scheduler.optimizer.param_groups[0]['lr'] #lr를 보려고 만듬 

        if e%10==0:#10epoch마다 출력합니다. 
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                e,result['train_loss'][e], valid_loss[e]))

            # print training/validation statistics 
            print('Epoch: {} \tTraining accuracy: {:.6f} \tValidation accuracy: {:.6f}'.format(
                e, train_acc[e], valid_acc[e]))
             #print learning_rate during training  
            print("current lr:  {}".format(scheduler.optimizer.param_groups[0]['lr']))
           
        if early_stop_mode ==True:
            if early_stop == 50: # early stop부분입니다. count를 하면서, 학습이 진행되지 않으면 break합니다.
                print(f"Overfitting! stop train model, epoch is {e}")
                break
        # save model if validation loss has decreased
        if valid_loss[e] <= valid_loss_min:
            early_stop = 0
    #         print("current lr:  {}".format(scheduler.optimizer.param_groups[0]['lr']))
    #         print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
    #             e+1, result['train_loss'][e], valid_loss[e]))
    #         print training/validation statistics 
    #         print('Epoch: {} \tTraining accuracy: {:.6f} \tValidation accuracy: {:.6f}'.format(
    #             e+1, train_acc[e], valid_acc[e]))
    ##validation loss가 줄어들 때 마다 print합니다. 
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,valid_loss[e]))
            torch.save(model.state_dict(), f'pt/{path}.pt')#validation loss가 줄어들면 save
            valid_loss_min = valid_loss[e]
    print("Finished_Training")
    idx = torch.argmin(valid_loss).item()
    i = torch.argmin(result['train_loss']).item()
    print('--------------\nepoch:{}\nbest result['train_loss']: train:{:.6f} valid:{:.6f}, '.format(i+1,result['train_loss'][i],valid_loss[i]))
    print('best train_acc: train:{:.3f} valid:{:.3f}, '.format(train_acc[i],valid_acc[i]))
    print('--------------\nepoch:{}\nbest valid_loss: train:{:.6f} valid:{:.6f}, '.format(idx+1,result['train_loss'][idx],valid_loss[idx]))
    print('best valid_acc: train:{:.3f} valid:{:.3f}, '.format(train_acc[idx],valid_acc[idx]))    
    
    #plot part 
    plot_loss(result['train_loss'],valid_loss,lr_list,path)
    
