function [a_out, Weight, Alpha, PrA] = ALCOVE_A_S(Exemplar,Sti,Target,Weight,Alpha,Parm)
% Attention Learning Covering Map (Kruschke, 1992) with attention learning
% Inputs:
%   Exemplar - exemplars in the psychological space
%   Sti - a new stimulus that is going to be added into the model; the
%         number of columns must be equal to that of the EX
%   Target - categories of the exemplars in the model (e.g., catergory A or
%            B)
%   Weight - associated weights between the hidden nodes and output nodes
%   Alpha - attention weight parameter
%   Parm - a 1-by-6 vector consists of 6 parameters: (1) psychological 
%          distance relationship (r=1, city-block metric; r=2, Euclidean 
%          metric); (2) exponential similarity gradient; (3) 
%          discriminability; (4) learning rate for associated weights; 
%          (5) learning rate for attentional weights; and (6) response 
%          mapping constant
% Outputs:
%   a_out - output node activation
%   Weight - updated assocated weights between hidden and output nodes
%   Alpha - updated attentional weights
%   PrA - probability of categorizing a stimulus to category A
% 
% Written by Chi-Hsun Eric Chang, Oct 2014, for a project in a course:
% The Application of Neural Network in Psychology

%%
D = abs(Exemplar-Sti);
a_hid = exp(-Parm(3).*((Alpha'*D.^Parm(1)).^(Parm(2)/Parm(1)))); % Hidden node activation, 1*hidden matrix

a_out = a_hid*Weight; % Output node activation, 1*output matrix
E = Target-a_out; % Error
delta_w = Parm(4).*(a_hid'*E); % Changed of association weights, hidden*output matrix
Weight = Weight+delta_w; % Associated weights update

EH = E*Weight'; % Sigma(k)E*weight, 1*hidden matrix
EJ = EH.*a_hid; % Note: not matrix multiplication, it's element-wise multiplicaiton
delta_a = -Parm(5).*(EJ*(Parm(3).*D')); % Changed of attetion weights, 1*dimension matrix
Alpha = Alpha+delta_a'; % Attention weights update, dimension*1 matrix
Alpha = Alpha/sum(Alpha); % Normalize

% Probability of categorizing stimulus to "A" category
a_out_sum = sum(exp(Parm(6).*a_out));
PrA = exp(Parm(6).*a_out(1))/a_out_sum;

end
