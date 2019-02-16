% Final Project for The Application of Neural Network in Psychology

%% Trial-by-trial GCM Model 
% One stimulus is added into the psychological space at a time to update 
% exemplars 
clear all

% Data
data = xlsread('stimulusSequences.csv');
sinu = data(:,3); % Stimulus of sinusoidal category
jump = data(:,4); % Stimulus of discrete jump category
sti = [sinu' jump']; % Combine stimuli of sinu and jump
% Category: Label 1 denotes sinu group; Label 2 denotes jump group
cate = cat(2,ones(1,length(sinu)),repmat(2,1,length(jump))); 
% Random sequence of stimuli
%ran = randperm(length(sti)); % Random trial label
ran = cat(2,(1:2:99),(2:2:100)); % Sequence: ABABAB...
Sti = zeros(1,length(sti));
Cate = zeros(1,length(sti));
for i = 1:length(sti)
    Sti(1,i) = sti(ran==i);
    Cate(1,i) = cate(ran==i);
end
% Parameters
NumSt = length(sti); % Number of stimulus
NumDim = 1; % Number of input dimension
a = repmat(1/NumDim,NumDim,1); % Alpha: Attention
b = 0.5; % Beta: Bias, 0.5 means no bias
r = 1; % r = 1 means city-block
c = 2; % Discriminability
par(1) = r;
par(2) = c;

% Probability of categorize a stimulus to category A
mPrA = zeros(NumSt,1); 
for t = 1:NumSt
    ex = Sti(:,1:t-1); % Exemplar
    st = repmat(Sti(:,t),1,t-1); % Stimulus
    ca = Cate(:,1:t-1); % Category label
    %if length(unique(ex)) == length(ex)
         %ex = Sti(:,1:t-1);
         %st = repmat(Sti(:,t),1,t-1);
         %ca = Cate(:,1:t-1);
    %else
        %[C,ia,ic] = unique(ex,'stable');
        %ex = C;
        %st = repmat(Sti(:,t),1,length(C));
        %ca = Cate(ia);
    %end
    [Sim,PrA] = GCMt(ex,st,ca,a,b,par);
    % If the similarity of category A and B are both zero, the probability equals 0.5
    if and(sum(Sim(ca==1))==0,sum(Sim(ca==2))==0) 
        mPrA(t,1) = 0.5;
    else
        mPrA(t,1) = PrA;
    end
end

% Probability of correct response
mPrc = zeros(NumSt,1);
for i = 1:NumSt
    if Cate(1,i) == 1
        mPrc(i,1) = mPrA(i,1);
    else
        mPrc(i,1) = 1-mPrA(i,1);
    end
end

% Plot
figure
plot(1:5:96,mPrc(1:5:96),'-^')
axis([0 100 0 1])
xlabel('Trial')
ylabel('Probability of Correct Response')

%% ALCOVE model, single updating
clear all

% Data
data = xlsread('stimulusSequences.csv'); % Load data
sinu = data(:,3); % Stimulus of sinusoidal category
jump = data(:,4); % Stimulus of discrete jump category
sti = [sinu' jump']; % Combine stimuli of sinu and jump
% Target: Label [1 -1] denotes sinu group; Label [-1 1] denotes jump group
targ = cat(1,repmat([1 -1],length(sinu),1),repmat([-1 1],length(jump),1));

% Random sequence of stimuli
%ran = randperm(length(sti)); % Random trial label
ran = cat(2,(1:2:99),(2:2:100)); % Sequence: ABABAB...
Sti = zeros(1,length(sti));
Targ = zeros(length(sti),2);
% Stimuli sequence: ABAB...
for i = 1:length(sti)
    Sti(1,i) = sti(ran==i);
    Targ(i,:) = targ(ran==i,:);
end

% Parameters
NumSt = length(sti); % Number of stimulus
NumDim = 1; % Number of input dimension
%NumHid = NumSt;
NumOut = 2; % Number of output(catecogy)
a = repmat(1/NumDim,NumDim,1); % Alpha: Attention
%w = rand(1,NumOut)-0.5; % Initial associated weight between hidden and output
w = zeros(1,NumOut); % Initial associated weight between hidden and output
r = 1; % r = 1 means city-block for separable psychological dimension
       % r = 2 means Euclidean metric for integral psychological dimension
q = 1; % q = 1 means exponential similarity gradient
c = 2; % Discriminability
lw = 0.03; % Learning rate for associated weights
la = 0.0033; % Learning rate for attentional weights
phi = 2; % Response mapping constant
par(1) = r;
par(2) = q;
par(3) = c;
par(4) = lw;
par(5) = la;
par(6) = phi;

% Probability of categorize a stimulus to category A(sinu)
mPrA = zeros(NumSt,1);
mPrA(1,1) = 0.5; % The first trial was guessed because there is no exemplar
for t = 2:NumSt
    ex = Sti(:,1:t-1); % Exemplar
    st = repmat(Sti(:,t),1,t-1); % Stimulus
    ta = Targ(t,:); % Target
    % Check whether there are repeatable exemplar
    if length(unique(ex)) == length(ex)
         ex = Sti(:,1:t-1);
         st = repmat(Sti(:,t),1,t-1);
    else
        [C,ia,ic] = unique(ex,'stable');
        ex = C;
        st = repmat(Sti(:,t),1,length(C));
    end
    [a_out, Weight, Alpha, PrA] = ALCOVE_A_S(ex,st,ta,w,a,par);
    %w = cat(1,Weight,(rand(1,2)-0.5));
    % Check whether current stimulus is the same with existed exemplar
    check = Sti(:,t) == Sti(:,1:t-1);
    if any(check) == 1
        w = Weight;
    else
        w = cat(1,Weight,zeros(1,NumOut));
    end
    a = Alpha; % Update attetional weight
    mPrA(t,1) = PrA; % Probability of categorize a stimulus to category A(sinu)
end

% Probability of correct response
mPrc = zeros(NumSt,1);
for i = 1:NumSt
    if Targ(i,1) == 1
        mPrc(i,1) = mPrA(i,1);
    else
        mPrc(i,1) = 1-mPrA(i,1);
    end
end

% Plot the probability of correct response
figure
plot(1:5:96,mPrc(1:5:96),'-^')
%plot(mPrc)
axis([0 100 0 1])
xlabel('Trial')
ylabel('Probability of Correct Response')

% Sort by category
Psinu = mPrc(Targ(:,1)==1); % Probability of correct response of "sinu" category
sti_sinu = Sti(Targ(:,1)==1); % Select the stimuli of sinu category
[B,IX] = sort(sti_sinu); % Sort stimuli in ascending way
% The probability of correct response of "sinu" category, presented
% following the order of stimuli's length
arr_sinu = zeros(length(sinu),1); 
for i = 1:length(sinu)
    arr_sinu(i,1) = Psinu(IX(i));
end

Pjump = mPrc(Targ(:,1)==-1); % Probability of correct response of "jump" category
sti_jump = Sti(Targ(:,1)==-1); % Select the stimuli of jump category
[B,IX] = sort(sti_jump); % Sort stimuli in ascending way
% The probability of correct response of "jump" category, presented
% following the order of stimuli's length
arr_jump = zeros(length(jump),1); 
for i = 1:length(jump)
    arr_jump(i,1) = Pjump(IX(i));
end
% Plot by category: X axis: stimulus; Y axis: probability of correct resoponse
figure
plot(arr_sinu,'r')
hold on
plot(arr_jump,'b')
axis([0 50 0 1])
legend('sinu','jump','Location','NorthOutside')
xlabel('Stimuli')
ylabel('Probability of Correct Response')

% Sort sitmuli by first and second present
[st1,ia1,ic1] = unique(Sti,'first'); % First present stimuli
first = zeros(1,NumSt/2); % The probability of correct response for first present stimuli
for i = 1:NumSt/2
    first(1,i) = mPrc(ia1(i));
end

[st2,ia2,ic2] = unique(Sti,'last'); % Second present stimuli
second = zeros(1,NumSt/2); % The probability of correct response for second present stimuli
for i = 1:NumSt/2
    second(1,i) = mPrc(ia2(i));
end
% Plot by first or second present: X axis: stimulus; Y axis: probability of correct resoponse
figure
plot(first,'r')
hold on
plot(second,'b')
axis([0 50 0 1])
legend('1st','2nd','Location','NorthOutside')
xlabel('Stimuli')
ylabel('Probability of Correct Response')

%% ALCOVE model, single updating and all exemplars exist at once
clear all

% Data
data = xlsread('stimulusSequences.csv');
sinu = data(:,3); % Stimulus of sinusoidal category
jump = data(:,4); % Stimulus of discrete jump category
sti = [sinu' jump']; % Combine stimuli of sinu and jump
% Target: Label [1 -1] denotes sinu group; Label [-1 1] denotes jump group
targ = cat(1,repmat([1 -1],length(sinu),1),repmat([-1 1],length(jump),1));

% Random sequence of stimuli
%ran = randperm(length(sti)); % Random trial label
ran = cat(2,(1:2:99),(2:2:100)); % Sequence: ABABAB...
Sti = zeros(1,length(sti));
Targ = zeros(length(sti),2);
for i = 1:length(sti)
    Sti(1,i) = sti(ran==i);
    Targ(i,:) = targ(ran==i,:);
end

% Parameters
NumSt = length(sti); % Number of stimulus
NumDim = 1; % Number of input dimension
%NumHid = NumSt;
NumOut = 2;
a = repmat(1/NumDim,NumDim,1); % Alpha: Attention
%w = rand(1,NumOut)-0.5; % Initial associated weight between hidden and output
w = zeros(NumSt/2,NumOut); % Associated weight, hidden*output matrix
r = 1; % r = 1 means city-block for separable psychological dimension
       % r = 2 means Euclidean metric for integral psychological dimension
q = 1; % q = 1 means exponential similarity gradient
c = 2; % Discriminability
lw = 0.03; % Learning rate for associated weights
la = 0.0033; % Learning rate for attentional weights
phi = 2; % Response mapping constant
par(1) = r;
par(2) = q;
par(3) = c;
par(4) = lw;
par(5) = la;
par(6) = phi;

% Probability of categorize a stimulus to category A(sinu)
mPrA = zeros(NumSt,1);
%mPrA(1,1) = 0.5; % The first trial was guessed because there is no exemplar
for t = 1:NumSt
    ex = unique(Sti);
    st = repmat(Sti(:,t),1,NumSt/2);
    ta = Targ(t,:);
    [a_out, Weight, Alpha, PrA] = ALCOVE_A_S(ex,st,ta,w,a,par);
    %w = cat(1,Weight,(rand(1,2)-0.5));
    w = Weight;
    a = Alpha;
    mPrA(t,1) = PrA;
end

% Probability of correct response
mPrc = zeros(NumSt,1);
for i = 1:NumSt
    if Targ(i,1) == 1
        mPrc(i,1) = mPrA(i,1);
    else
        mPrc(i,1) = 1-mPrA(i,1);
    end
end

% Plot the probability of correct response
figure
plot(1:5:96,mPrc(1:5:96),'-^')
%plot(mPrc)
axis([0 100 0 1])
xlabel('Trial')
ylabel('Probability of Correct Response')

% Plot the probability of correct response of sinu category
figure
subplot(2,1,2)
plot(1:2:99,mPrc(1:2:99),'-o')
axis([0 100 0 1])
title('sinu')
xlabel('Trial')
ylabel('Probability of Correct Response')
% Plot the probability of correct response of jump category
%figure
subplot(2,1,1)
plot(2:2:100,mPrc(2:2:100),'-s')
axis([0 100 0 1])
title('jump')
xlabel('Trial')
ylabel('Probability of Correct Response')

%% ALCOVE model, single updating, double number of stimuli 
clear all

% Data
data = xlsread('stimulusSequences.csv');
sinu = data(:,3); % Stimulus of sinusoidal category
jump = data(:,4); % Stimulus of discrete jump category
sti = [sinu' jump']; % Combine stimuli of sinu and jump
% Target: Label [1 -1] denotes sinu group; Label [-1 1] denotes jump group
targ = cat(1,repmat([1 -1],length(sinu),1),repmat([-1 1],length(jump),1));

% Random sequence of stimuli
%ran = randperm(length(sti)); % Random trial label
ran = cat(2,(1:2:99),(2:2:100)); % Sequence: ABABAB...
Sti = zeros(1,length(sti));
Targ = zeros(length(sti),2);
for i = 1:length(sti)
    Sti(1,i) = sti(ran==i);
    Targ(i,:) = targ(ran==i,:);
end
Sti = [Sti Sti]; % Double the stimulus
Targ = [Targ;Targ];

% Parameters
NumSt = length(sti)*2; % Number of stimulus
NumDim = 1; % Number of input dimension
%NumHid = NumSt;
NumOut = 2;
a = repmat(1/NumDim,NumDim,1); % Alpha: Attention
%w = rand(1,NumOut)-0.5; % Initial associated weight between hidden and output
w = zeros(NumSt/4,NumOut); % Associated weight, hidden*output matrix
r = 1; % r = 1 means city-block for separable psychological dimension
       % r = 2 means Euclidean metric for integral psychological dimension
q = 1; % q = 1 means exponential similarity gradient
c = 2; % Discriminability
lw = 0.03; % Learning rate for associated weights
la = 0.0033; % Learning rate for attentional weights
phi = 2; % Response mapping constant
par(1) = r;
par(2) = q;
par(3) = c;
par(4) = lw;
par(5) = la;
par(6) = phi;

% Probability of categorize a stimulus to category A(sinu)
mPrA = zeros(NumSt,1);
%mPrA(1,1) = 0.5; % The first trial was guessed because there is no exemplar
for t = 1:NumSt
    ex = unique(Sti);
    st = repmat(Sti(:,t),1,NumSt/4);
    ta = Targ(t,:);
    [a_out, Weight, Alpha, PrA] = ALCOVE_A_S(ex,st,ta,w,a,par);
    %w = cat(1,Weight,(rand(1,2)-0.5));
    w = Weight;
    a = Alpha;
    mPrA(t,1) = PrA;
end

% Probability of correct response
mPrc = zeros(NumSt,1);
for i = 1:NumSt
    if Targ(i,1) == 1
        mPrc(i,1) = mPrA(i,1);
    else
        mPrc(i,1) = 1-mPrA(i,1);
    end
end

% Plot the probability of correct response of sinu category
figure
subplot(2,1,2)
plot(1:2:199,mPrc(1:2:199),'-o')
axis([0 200 0 1])
title('sinu')
xlabel('Trial')
ylabel('Probability of Correct Response')
% Plot the probability of correct response of jump category
%figure
subplot(2,1,1)
plot(2:2:200,mPrc(2:2:200),'-s')
axis([0 200 0 1])
title('jump')
xlabel('Trial')
ylabel('Probability of Correct Response')

%% ALCOVE model, single updating, all exemplars exist at once, double number of stimuli 
clear all

% Data
data = xlsread('stimulusSequences.csv');
sinu = data(:,3); % Stimulus of sinusoidal category
jump = data(:,4); % Stimulus of discrete jump category
sti = [sinu' jump']; % Combine stimuli of sinu and jump
% Target: Label [1 -1] denotes sinu group; Label [-1 1] denotes jump group
targ = cat(1,repmat([1 -1],length(sinu),1),repmat([-1 1],length(jump),1));

% Random sequence of stimuli
%ran = randperm(length(sti)); % Random trial label
ran = cat(2,(1:2:99),(2:2:100)); % Sequence: ABABAB...
Sti = zeros(1,length(sti));
Targ = zeros(length(sti),2);
for i = 1:length(sti)
    Sti(1,i) = sti(ran==i);
    Targ(i,:) = targ(ran==i,:);
end
Sti = [Sti Sti]; % Double the stimulus
Targ = [Targ;Targ];

% Parameters
NumSt = length(sti)*2; % Number of stimulus
NumDim = 1; % Number of input dimension
%NumHid = NumSt;
NumOut = 2;
a = repmat(1/NumDim,NumDim,1); % Alpha: Attention
%w = rand(1,NumOut)-0.5; % Initial associated weight between hidden and output
w = zeros(1,NumOut);
r = 1; % r = 1 means city-block for separable psychological dimension
       % r = 2 means Euclidean metric for integral psychological dimension
q = 1; % q = 1 means exponential similarity gradient
c = 2; % Discriminability
lw = 0.03; % Learning rate for associated weights
la = 0.0033; % Learning rate for attentional weights
phi = 2; % Response mapping constant
par(1) = r;
par(2) = q;
par(3) = c;
par(4) = lw;
par(5) = la;
par(6) = phi;

% Probability of categorize a stimulus to category A(sinu)
mPrA = zeros(NumSt,1);
mPrA(1,1) = 0.5; % The first trial was guessed because there is no exemplar
for t = 2:NumSt
    ex = Sti(:,1:t-1);
    st = repmat(Sti(:,t),1,t-1);
    ta = Targ(t,:);
    if length(unique(ex)) == length(ex)
         ex = Sti(:,1:t-1);
         st = repmat(Sti(:,t),1,t-1);
    else
        [C,ia,ic] = unique(ex,'stable');
        ex = C;
        st = repmat(Sti(:,t),1,length(C));
    end
    [a_out, Weight, Alpha, PrA] = ALCOVE_A_S(ex,st,ta,w,a,par);
    %w = cat(1,Weight,(rand(1,2)-0.5));
    check = Sti(:,t) == Sti(:,1:t-1);
    if any(check) == 1
        w = Weight;
    else
        w = cat(1,Weight,zeros(1,NumOut));
    end
    a = Alpha;
    mPrA(t,1) = PrA;
end

% Probability of correct response
mPrc = zeros(NumSt,1);
for i = 1:NumSt
    if Targ(i,1) == 1
        mPrc(i,1) = mPrA(i,1);
    else
        mPrc(i,1) = 1-mPrA(i,1);
    end
end

% Plot the probability of correct response of sinu category
figure
subplot(2,1,2)
plot(1:2:199,mPrc(1:2:199),'-o')
axis([0 200 0 1])
title('sinu')
xlabel('Trial')
ylabel('Probability of Correct Response')
% Plot the probability of correct response of jump category
%figure
subplot(2,1,1)
plot(2:2:200,mPrc(2:2:200),'-s')
axis([0 200 0 1])
title('jump')
xlabel('Trial')
ylabel('Probability of Correct Response')





