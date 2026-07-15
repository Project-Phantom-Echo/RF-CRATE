function varargout = myacf(y,numLags)
%AUTOCORR Sample autocorrelation


if numLags > (sum(~isnan(y))-1)
  
    error(message('econ:autocorr:LagsTooLarge'))

end

% Preprocess validated inputs:
y         = y(:);
y         = y - mean(y,"omitnan");
N         = sum(~isnan(y)); % Effective sample size


if N < length(y) % Missing data

% Compute ACF in the presence of missing data.
%
%   To compute the jth autocovariance, compute the cross product of y(t)
%	and y(t+j) with sample size (T) adjusted for missing data. The 
%	autocovariances computed in the presence of missing data follow
%	expression 2.1.10 on page 31 of [1].
%
%	The procedure assumes data is missing at random. Asymptotically, if P
%	is the probability that any element of y(t) is not missing, then for
%	all j > 0:
%
%	sum(~isnan(cross))/length(y) --> P^2,
%
%   or
%
%	length(y)/(T-sum(~isnan(y(1:j)))) --> 1/P^2.
%
%	If the input series y(t) has no missing data, time-domain
%	autocovariances are identical to those in the FFT approach.

	acf = nan(numLags+1,1);

    for j = 0:numLags
        
        cross   = y(1:end-j).*y(j+1:end);
        iNonNaN = ~isnan(cross);

        if any(iNonNaN)
            
            T        = sum(iNonNaN)+sum(~isnan(y(1:j)));
            acf(j+1) = sum(cross,"omitnan")/T;
            
        end
        
    end

else % No missing data

% Compute ACF using expression 2.1.10 on page 31 of [1], but perform the
% computation in the frequency domain using an FFT:

    nFFT = 2^(nextpow2(length(y))+1);
    F = fft(y,nFFT);
    F = F.*conj(F);
    acf = ifft(F);
    acf = acf(1:(numLags+1)); % Retain nonnegative lags
    acf = real(acf);
   
end

acf = acf./acf(1); % Normalize

lags = (0:numLags)';

varargout = {acf,lags};