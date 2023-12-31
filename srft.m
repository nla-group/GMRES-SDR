function S = srft(n, s)

	% S = srft(n, s)
	%* Generates a Subsampled Random Cosine Transform, to be used as a random subspace embedding
	% SS = sqrt(n/s) * D * F * E, where:
	%	D = random projection on s coordinates
	%	F = discrete cosine transform
	%	E = random sign diagonal matrix
	% n is the original space size
	% s is the embedding dimension
	% Output:
	%	S: function handle such that S(x) = SS*x, where SS is the s x n matrix corresponding to the randomized embedding

	sgn = 2*randi(2, n, 1) - 3;
	p = randperm(n, s);
	S1 = @(x) dct(sgn.*x);
    %S1 = @(x) real(fft(sgn.*x));
    %S1 = @(x) x;
	S2 = @(x) x(p, :);
	S = @(x) sqrt(n/s) * S2(S1(x));
end