#손실함수 그리기
plt.figure(figsize=(20,10))
plt.plot(total_loss)
plt.show()
#주가그리기
concatdata=torch.utils.data.ConcatDataset([train_dataset,test_dataset])
data_loader=torch.utils.data.DataLoader(dataset=concatdata,batch_size=100)
with torch.no_grad():
    pred=[]
    net.eval()
    for data in data_loader:
        seq,target=data
        out=net(seq)
        pred+=out.cpu().tolist()
plt.figure(figsize=(20,10))

plt.plot(price['total_price'][sequence_length:].values,'--')
plt.plot(pred,'b',linewidth=0.6)
plt.legend(['actual','prediction'])
plt.show()
