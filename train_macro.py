#train.py 파일을 macro로 간편하게 사용할 수 있도록 만들어둔 템플릿
def train_macro(model,epoch,train_loader,valid_loader,set_optimizer,path):
    n_epochs = epoch
#     set_optimizer=set_optim
    if set_optimizer ==0:
        optimizer = optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2)
    #     optimizer = optim.AdamW(model.parameters(),lr= 1e-3)transfer learning이 아닐 때 
        scheduler = cosine.CosineAnnealingWarmupRestarts(optimizer,
                                                first_cycle_steps=100,
                                                cycle_mult=1,
                                                max_lr=1e-1,
                                                min_lr=1e-3,
                                                warmup_steps=50,
                                                gamma=0.5)
    elif set_optimizer==1:
        optimizer = optim.SGD(filter(lambda p:p.requires_grad,model.parameters()),lr= 1e-2,momentum=0.9,weight_decay=1e-8)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode = 'min',patience=10,factor = 0.1)
    elif set_optimizer==2:
        optimizer = optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),lr= 1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode = 'min',patience=10,factor = 0.1)

# filter(lambda p:p.requires_grad,model.parameters())
#pretrain할 때 좋음, requires_grad가 true인것만 optimizer에 들어가서 학습율을 빨리함
    criterion=nn.CrossEntropyLoss()
    train(model,n_epochs=n_epochs,set_optimizer=set_optimizer,optimizer=optimizer,scheduler=scheduler,
      early_stop_mode=False,criterion=criterion,train_loader=train_loader, valid_loader=valid_loader ,path= path)
