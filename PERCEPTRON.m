classdef PERCEPTRON < handle
    %PERCEPTRON Implements a multi-layer perceptron with sigmoid tfn
    % Copyright Anton Tkachev 2015
    %% Properties of the perceptron
    properties
        layer;    % array that defines the number of neurons in each layer
        nLayers;  % total number of layers  
        nTrans;   % total number of transitions between the layers
        weight;   % cell of neurons' weights matrices
        alpha;    % sigmoid function coefficient
    end
    
    %% Methods of the perceptron
    methods
        %% Constructor
        function obj = PERCEPTRON(layers_vector)
            obj.alpha = 2.5;
            obj.layer = layers_vector;
            obj.nLayers = length(layers_vector);
            obj.nTrans = length(layers_vector) - 1;
            obj.weight = cell(obj.nTrans,1);
            
            a = 0.1;    % bounds for weights random initialization
            for i = 1 : obj.nTrans
                obj.weight{i} = random('Uniform',-a,a,obj.layer(i+1),obj.layer(i));
            end
        end
        
        %% Forward neural network calculation
        function out = forward(obj,input_col_vector)
            A = cell(obj.nLayers,1);
            A{1} = input_col_vector;
            for i = 1 : obj.nTrans
                A{i+1} = PERCEPTRON.tfn(obj.weight{i}*A{i},obj.alpha);
            end
            out = A{obj.nLayers};
        end
        
        %% Error back propagation. Single iteration
        function err = backprop(obj,input,desired_output,eta)
            O = cell(obj.nTrans + 1,1);
            O{1} = input;
            for i = 1 : obj.nTrans
                O{i+1} = PERCEPTRON.tfn(obj.weight{i}*O{i},obj.alpha);
            end
            O = flip(O);
            T = desired_output;
            N = flip(obj.layer);
            W = flip(obj.weight);
            delta = cell(obj.nTrans,1);
            
            delta{1} = -2*obj.alpha*O{1}.*(ones(N(1),1) - O{1}).*(T - O{1});
            err = (T - O{1});
            
            for i = 2 : obj.nTrans
                delta{i} = 2*obj.alpha*O{i}.*(ones(N(i),1) - O{i}).*(W{i-1}.'*delta{i-1});
            end
            
            for i = 1 : obj.nTrans
                W{i} = W{i} - eta*delta{i}*O{i+1}.';
            end
            obj.weight = flip(W);
        end
    end
    
    %% Transfer function methods
    methods(Static, Access = private)
        %% Exponential sigmoid transfer function
        function out = tfn(input_vector,input_alpha)
            n = length(input_vector);
            out = zeros(n,1);
            for i = 1 : n
                out(i) = 1/(1 + exp(-2*input_alpha*input_vector(i)));
            end
        end
    end
    
end
