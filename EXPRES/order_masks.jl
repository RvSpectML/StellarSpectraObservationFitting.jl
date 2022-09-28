if order==10
    SSOF.mask_telluric_feature!(data, log(4030), log(4035))
    SSOF.mask_telluric_feature!(data, log(4076.9), log(4081))
end

if order==11
    SSOF.mask_telluric_feature!(data, log(4057), log(4065))
    SSOF.mask_telluric_feature!(data, log(4100), log(4110))
end

if order==12
    SSOF.mask_telluric_feature!(data, log(4083), log(4086.5))
    SSOF.mask_telluric_feature!(data, log(4134), log(4136))
end

if order==14
    SSOF.mask_telluric_feature!(data, log(4139), log(4146))
    SSOF.mask_telluric_feature!(data, log(4186), log(4192))
end
