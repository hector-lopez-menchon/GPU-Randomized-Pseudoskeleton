function [Cp,Up,Rp] = postcompress_wrapper(C,U,R,tol)
        T.u = C; T.s = U; T.v = R;
        z = svd_compress(T,tol);
        Cp = z.u; Up = z.s; Rp = z.v;
end




