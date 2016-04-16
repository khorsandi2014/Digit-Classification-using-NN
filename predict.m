function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);


p = zeros(size(X, 1), 1);
% Add ones to the X data matrix
X = [ones(m, 1) X];

    z2 = Theta1*X';
    a2 = [ones(m,1) sigmoid(z2')];
    z3 = Theta2 * a2';
    A = sigmoid(z3');
    [value p] = max ( A , [] , 2);

end
