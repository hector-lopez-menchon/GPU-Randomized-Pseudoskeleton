function [vertex, topol,trian,edges, un, ds, ln, cent, N] = object_wrapper_spheres_interaction2(lambda,radius,d,Ne_input)

        %Permite espepcificar los parámetros
        %lambda = 1;
        k = 2*pi/lambda;
        eta = 120*pi;
        field = 1; % EFIE->1, MFIE->2, CFIE->3

        Rint_s = 1;       % MoM Integration radius (meters). Rint=0 is enough if basis functions are very small.
        Rint_f = Rint_s;
        Ranal_s = 1;
        corr_solid = 0;
        flag = 0;

        EM_data = struct('lambda',lambda, 'k',k, 'eta',eta, 'field',field, 'Rint_s',Rint_s, 'Rint_f',Rint_f, 'Ranal_s',Ranal_s, 'corr_solid',corr_solid, 'flag',flag );


        cd objects
        obj_1 = sphere(struct('R',radius,'Ne',Ne_input));

        cd ..
        Nv1 = size(obj_1.vertex,2); Nt1 = size(obj_1.topol,2);

        cd objects
        obj_2 = sphere(struct('R',radius,'Ne',Ne_input));
        cd ..
        Nv2 = size(obj_2.vertex,2); Nt2 = size(obj_2.topol,2);

        %Object translation
        [mmm,nnn] = size(obj_2.vertex);
        obj_2.vertex(1,:) = obj_2.vertex(1,:) + d;

        % Join objects
        obj.vertex = [obj_1.vertex obj_2.vertex];
        obj.topol  = [obj_1.topol  obj_2.topol+Nv1];
        n1 = 1 : Nt1;           % Indices to object 1 topol
        n2 = Nt1+1 : Nt1+Nt2;   % Indices to object 2 topol
        obj = get_edge(obj);
        number_edges = size(obj.edges,2);


        vertex = obj.vertex;
        topol = obj.topol;
        trian = obj.trian;
        edges = obj.edges;
        un = obj.un;
        ds = obj.ds;
        ln = obj.ln;
        cent = obj.cent;
        N = number_edges;



end





