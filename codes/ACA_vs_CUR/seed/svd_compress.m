

function z = svd_compress(x,thresh)

% uses SVD to compress s

if ischar(x), x=readfile(x,0); end

if isnumeric(x), [u,s,v] = svd(x,'econ');
    ss = diag(s);
    ss = ss/max(ss);
    drsi = find(ss > thresh);
    z.u = u(:,drsi);
    z.s = s(drsi,drsi);
    z.v = v(:,drsi)';
    return
end

if isstruct(x) && isfield(x,'u')
    [qu,ru] = qr_cbd(x.u);
    [qva,rva] = qr_cbd(x.v');
    s = ru*x.s*rva';
    [u,s,v] = svd(s,'econ');
    drs = diag(s);
    drs=drs/max(drs);
    drsi = find(drs > thresh);
    u = u(:,drsi);
    s = s(drsi,drsi);
    v = v(:,drsi);
    z.u = qu*(u);
    z.s = s;
    z.v = (qva*v)';
    return
end




