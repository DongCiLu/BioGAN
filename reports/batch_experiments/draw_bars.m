raw = raw_accuracy;
trans = trans_accuracy;
ylabel_string = 'Validation accuracy';

legend_string = {'Without Transfer Learning' 'With Transfer Learning'};
n = size(raw, 2);
mean_acc = [mean(raw); mean(trans)]; % mean accuracy
std_acc = [std(raw); std(trans)];  % standard deviation of accuracy
figure
hold on
hb = bar(1:n, mean_acc');
% For each set of bars, find the centers of the bars, and write error bars
pause(0.1); %pause allows the figure to be created
for ib = 1:numel(hb)
    %XData property is the tick labels/group centers; XOffset is the offset
    %of each distinct group
    xData = hb(ib).XData+hb(ib).XOffset;
    errorbar(xData,mean_acc(ib,:),std_acc(ib,:),'k.')
end

set(gca,'FontSize',20);
set(gca,'XTick',(1:n));
% set(gca,'XTickLabel',{'100%'; '90%'; '50%'; '20%'; '10%'; '5%'});
set(gca,'XTickLabel',{'100%'; '90%'; '80%'; '70%'; '60%'; '50%'; '40%'; '30%'; '20%'; '10%'});
xlabel_hand=xlabel('Training set percentage');
set(xlabel_hand,'Fontname', 'Times New Roman', 'Fontsize', 20);
ylabel_hand=ylabel(ylabel_string);
set(ylabel_hand,'Fontname', 'Times New Roman', 'Fontsize', 20);
legend_hand = legend(legend_string(1:2));
set(legend_hand,'Fontname', 'Times New Roman', 'Fontsize', 20);
