if order==10
    SSOF.mask_tellurics!(data, log(4030), log(4035))
    SSOF.mask_tellurics!(data, log(4076.9), log(4081))
end
