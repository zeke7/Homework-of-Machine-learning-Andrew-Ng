function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1

X = [ones(m,1) X];

theta1_total = sigmoid(X * Theta1');

theta1_total = [ones(m,1) theta1_total]; %5000*26 a1
theta2_total = sigmoid(theta1_total * Theta2'); %5000*10 a2

h_theta = theta2_total;

y_k = zeros(m,num_labels); 

for i=1:m
    y_k(i,y(i,1)) = 1;
end

temp_J = zeros(m,num_labels);

for i=1:m
    for k=1:num_labels
        temp_J(i,k) = y_k(i,k) * log(h_theta(i,k)) + (1 - y_k(i,k)) * log(1-h_theta(i,k));
    end
end

J = (-1/m)*sum(temp_J(:));

regTheta1 =  Theta1(:,2:end);
regTheta2 =  Theta2(:,2:end);
regularization  = (lambda/(2*m)) * (sum(sum(regTheta1.^2)) + sum(sum(regTheta2.^2)));

J = J + regularization ;

% Part 2
   D_1 = zeros();
   D_2 = zeros();

   a_1 =  X; % 5000x401
   
   z_2 = a_1 * Theta1'; % 5000x25  
   z2 = [ones(m,1) z_2];
   a_2 = sigmoid(z_2); 
   a_2 = [ones(m,1) a_2]; % 5000x26
   
   z_3 = a_2 * Theta2'; 
   a_3 = sigmoid(z_3); % 5000*10
   
for t = 1:m
   a1 = a_1(t,:); % 1x401
   a2 = a_2(t,:); % 1x26
   a3 = a_3(t,:); % 1x10
   
   y = y_k(t,:); % 1x10
   delta3 = a3 - y; %1x10
   delta2 = Theta2' * delta3' .* sigmoidGradient(z2(t,:)');
    
   D_1 = D_1 + delta2(2:end) * a1;
   D_2 = D_2 + delta3' * a2;
   
end


Theta1_grad = (1/m) * D_1 + (lambda / m) * ([zeros(size(Theta1, 1),1) Theta1(:,2:end)]);
Theta2_grad = (1/m) * D_2 + (lambda / m) * ([zeros(size(Theta2, 1),1) Theta2(:,2:end)]);



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
