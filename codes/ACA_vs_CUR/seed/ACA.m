% [U,V] = ACA(ACA_thres, m,n, OG_data,EM_data)
% Adaptive Cross Approximation (ACA) matrix compression
%
% Input:
% ACA_thres = Relative error threshold to stop adding rows and columns in ACA iteration
% m, n      = Row and column indices of Z submatrix to compress
% OG_data   = struct containig data passed to user_impedance() function
% EM_data   = struct containig data passed to user_impedance() function
%
% Output:
% U*V = user_impedance(m,n, OG_data,EM_data), except for the approximation error (ACA_thres)
%
% ACA algorithm as described in (with some optimizations):
% Kezhong Zhao,Marinow Vouvakis and Jin-Fa Lee, “The Adaptive Cross Approximation
% Algorithm for Accelerated Method of Moment Computations of EMC Problems”,
% IEEE Transactions on Electromagnetic Compatibility, Vol. 47, No. 4, November 2005.
%
% Juan M. Rius, Jose M. Tamayo, AntennaLab, Universitat Politecnica de Catalunya (Spain), v1.0, August 2007

% /***************************************************************************
%  *   Copyright (C) 2007 by Juan M. Rius and Jose M. Tamayo                 *
%  *   AntennaLab, Universitat Politecnica de Catalunya, rius@tsc.upc.edu    *
%  *                                                                         *
%  *   This program is free software; you can redistribute it and/or modify  *
%  *   it under the terms of the GNU General Public License as published by  *
%  *   the Free Software Foundation; either version 2 of the License, or     *
%  *   (at your option) any later version.                                   *
%  *                                                                         *
%  *   This program is distributed in the hope that it will be useful,       *
%  *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
%  *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
%  *   GNU General Public License for more details.                          *
%  *                                                                         *
%  *   You should have received a copy of the GNU General Public License     *
%  *   along with this program; if not, write to the                         *
%  *   Free Software Foundation, Inc.,                                       *
%  *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
%  ***************************************************************************/

function [U,V] = ACA(ACA_thres, m,n, OG_data,EM_data)

M = length(m);
N = length(n);

% If Z is a vector, there is nothing to compress
if M==1 || N==1, U = user_impedance(m,n,OG_data,EM_data); V = 1; return
end

J = zeros(N,1); % Indices of columns picked up from Z
I = zeros(M,1); % Indices of rows picked up from Z
i = (2:M); % Row indices to search for maximum in R 
j = (1:N); % Column indices to search for maximum in R

%% Initialization

% Initialize the 1st row index I(1) = 1
I(1) = 1;

% Initialize the 1st row of the approximate error matrix
Rik = user_impedance(m(I(1)),n,OG_data,EM_data);

% Find the 1st column index J(1)
col = find( abs(Rik(j)) == max(abs(Rik(j))) );
J(1) = j(col(1));
j = remove_index(j,J(1));

% First row of V
V = Rik/Rik(J(1));

% Initialize the 1st column of the approximate error matrix
Rjk = user_impedance(m,n(J(1)),OG_data,EM_data);

% First column of U
U = Rjk;

% Norm of (approximate) Z, to test error
normZ = norm(U)^2 * norm(V)^2;

% Find 2nd row index I(2)
row = find( abs(Rjk(i)) == max(abs(Rjk(i))) );
I(2) = i(row(1));
i = remove_index(i,I(2));

% Iteration
for k=2:min(M,N)
    % Update (Ik)th row of the approximate error matrix:
    Rik = user_impedance(m(I(k)),n,OG_data,EM_data) - U(I(k),:)*V;

    % Find kth column index Jk
    col = find(abs(Rik(j)) == max(abs(Rik(j))) );
    J(k) = j(col(1));
    j = remove_index(j,J(k));
    
    % Terminate if R(I(k),J(k)) == 0
    if(Rik(J(k)) == 0) 
        break; 
    end
    
    % Set k-th row of V equal to normalized error
    Vk = Rik/Rik(J(k));
    
    % Update (Jk)th column of the approximate error matrix
    Rjk = user_impedance(m,n(J(k)),OG_data,EM_data) - U*V(:,J(k));
    
    % Set k-th column of U equal to updated error
    Uk = Rjk;

    % Norm of approximate Z
%    normZ = normZ + 2*sum(real((U'*Uk).*(Vk*V').')) + norm(Uk)^2*norm(Vk)^2;
    normZ = normZ + 2*sum(abs((U'*Uk).*(Vk*V').')) + norm(Uk)^2*norm(Vk)^2;
    
    % Update U and V
    U = [U Uk]; V = [V; Vk];
    
    % Check convergence
    if norm(Uk)*norm(Vk) <= ACA_thres*sqrt(normZ)
        break
    end
    
    if k==min(M,N) 
        break; 
    end
    
    % Find next row index
    row = find( abs(Rjk(i)) == max(abs(Rjk(i))) );
    I(k+1) = i(row(1));
    i = remove_index(i,I(k+1));
end

function i2 = remove_index(i,e)
N = length(i);
i2 = zeros(N-1,1);
row = find(i==e);
i2(1:row(1)-1) = i(1:row(1)-1);
i2(row(1):N-1) = i(row(1)+1:N);