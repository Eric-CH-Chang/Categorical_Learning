function [Sim,PrA] = GCMt(EX,ST,Cate,Alpha,Beta,Parm)
% Trial-by-trial Generalized Context Model (Nosofsky, 1986)
% Inputs:
%   EX - exemplars in the psychological space
%   ST - a new stimulus that is going to be added into the psychological 
%        space; the number of columns must be equal to that of the EX
%   Cate - categories of the exemplars in the psycholgoical space (e.g., 
%          catergory A or B)
%   Alpha - attention weight parameter
%   Beta - bais parameter
%   Parm - a 1-by-2 vector, psychological distance relationship (r=1, city-
%          block metric; r=2, Euclidean metric), and discriminability (c)
% Outputs:
%   Sim - similarity between stimuli and exemplars
%   PrA - probability of categorizing a stimulus to category A
% 
% Written by Chi-Hsun Eric Chang, Oct 2014, for a project in a course:
% The Application of Neural Network in Psychology

%%
dis = abs(EX-ST); % Exemplar minus Stimulus
D = Parm(2).*((Alpha'*dis.^Parm(1)).^(1/Parm(1))); % Distance between stimulus and all exemplars in space
Sim = exp(-D); % Similarity between stimulus and exemplars
PrA = (Beta*sum(Sim(Cate==1)))/((Beta*sum(Sim(Cate==1)))+((1-Beta)*sum(Sim(Cate==2)))); % Probability of categorizing stimulus to category A

end
