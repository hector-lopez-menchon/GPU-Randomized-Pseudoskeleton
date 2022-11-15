% view_var(text,x)
% View nicely the contents of variable with prefix text
%
% Juan M. Rius, AntennaLab, Universitat Politecnica de Catalunya (Spain), v1.0, October 1996

% /***************************************************************************
%  *   Copyright (C) 2007 by Juan M. Rius                                    *
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

function view_var(text,x)

st = [text, ' = '];

if isstr(x),
	disp([st x]);
	return
end

[N,M] = size(x);

stl = length(st);

if N*M>1,
	st = [st, '['];
	maxel = zeros(1,M-1);
	for n=1:N,
		for m=1:M-1,
			tmp = length(num2str(x(n,m)));
			if tmp > maxel(m), maxel(m) = tmp; end
		end
	end

	for n=1:N,
		for m=1:M-1,
			stel = num2str(x(n,m));
			st = [st, ' ', stel];

			for i=length(stel):maxel(m),
				st = [st, ' '];
			end
		end

		st = [st, ' ', num2str(x(n,M))];
		st = [st, sprintf('\n')];

		if n~=N, for i=1:stl+1,
			st = [st, ' '];
		end; end
	end
	st = [st(1:length(st)-1), ' ]'];
else
	st = [st, num2str(x)];
end
disp(st);

% num2str is necessary for displaying complex numbers
