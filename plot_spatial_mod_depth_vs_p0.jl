# Plot spatial modulation depth versus p0 (Julia version)
# Matches the plotting style used in threshold_growth.jl

using FFTW
using PyPlot
using Printf
using Statistics
using DelimitedFiles

const PEAK_PROMINENCE_FACTOR = 3.0
const MIN_PEAK_HEIGHT = 1e-8

function parse_args()
    filename = nothing
    xpos = nothing
    csv_path = nothing

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "-f" || arg == "--filename"
            i += 1
            filename = ARGS[i]
        elseif arg == "-x" || arg == "--xpos"
            i += 1
            xpos = parse(Float64, ARGS[i])
        elseif arg == "--csv"
            if i < length(ARGS) && !startswith(ARGS[i + 1], "-")
                i += 1
                csv_path = ARGS[i]
            else
                csv_path = "spatial_mod_depth_vs_p0.csv"
            end
        elseif arg == "-h" || arg == "--help"
            println(
                "Usage: julia plot_spatial_mod_depth_vs_p0.jl -f <input_dir_name> -x <x_position> [--csv [csv_path]]"
            )
            exit(0)
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end

    if filename === nothing || xpos === nothing
        error("Missing required arguments. Use -f <input_dir_name> and -x <x_position>.")
    end

    return filename, xpos, csv_path
end

function read_input(seed_path)
    values = Float64[]
    for line in eachline(seed_path)
        stripped = split(line, "!")[1]
        token = strip(stripped)
        if isempty(token)
            continue
        end
        if startswith(token, ".")
            continue
        end
        push!(values, parse(Float64, token))
    end

    if length(values) < 13
        error("seed.in does not contain the expected number of numeric entries.")
    end

    nodes = Int(values[1])
    return (
        nodes,
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
        values[8],
        values[9],
        values[10],
        values[11],
        values[12],
        Int(values[13])
    )
end

function extract_float(path)
    m = match(r"psi\d+_([\d.e-]+)", path)
    if m === nothing
        return 0.0
    end
    num_str = rstrip(m.captures[1], '.')
    return parse(Float64, num_str)
end

function load_data(output_dir)
    files = readdir(output_dir)
    psi_files = [joinpath(output_dir, f) for f in files if startswith(f, "psi")]
    return sort(psi_files, by=extract_float)
end

function find_temporal_cut_of_x_peaks(psi_data, starting_x_index0)
    col_index = starting_x_index0 + 1
    max_amp = maximum(psi_data[:, col_index])
    println("max_amp = $max_amp")

    x_centre_indices = Int[col_index]
    nrows, ncols = size(psi_data)
    for i in 2:nrows
        x_cut_data = view(psi_data, i, :)
        peak_indices = Int[]
        for j in 2:(ncols - 1)
            if x_cut_data[j] > x_cut_data[j - 1] && x_cut_data[j] > x_cut_data[j + 1]
                if x_cut_data[j] >= max_amp / PEAK_PROMINENCE_FACTOR
                    push!(peak_indices, j)
                end
            end
        end

        if !isempty(peak_indices)
            prev = x_centre_indices[end]
            diffs = abs.(peak_indices .- prev)
            _, min_idx = findmin(diffs)
            current_x_centre_index = peak_indices[min_idx]
        else
            current_x_centre_index = x_centre_indices[end]
        end

        push!(x_centre_indices, current_x_centre_index)
    end

    temporal_cut = [psi_data[i, x_centre_indices[i]] for i in 1:length(x_centre_indices)]
    println(temporal_cut)
    return temporal_cut
end

function find_p0_vals_from_filenames(sorted_files)
    pattern = r"(\d+(?:\.\d+)?e[-+]\d+)"
    p0_vals = Float64[]
    for file in sorted_files
        m = match(pattern, file)
        if m === nothing
            push!(p0_vals, 0.0)
        else
            push!(p0_vals, parse(Float64, m.captures[1]))
        end
    end
    return p0_vals
end

function fftfreq(n, d)
    val = 1.0 / (n * d)
    freq = zeros(Float64, n)
    nby2 = floor(Int, n / 2)
    for i in 0:(n - 1)
        if i <= nby2
            freq[i + 1] = i * val
        else
            freq[i + 1] = (i - n) * val
        end
    end
    return freq
end

function find_first_harmonic(fft_data, fft_freq_data; prominence=nothing, minimum_height=MIN_PEAK_HEIGHT)
    n = length(fft_data)
    peaks = Int[]
    for i in 2:(n - 1)
        if fft_data[i] > fft_data[i - 1] && fft_data[i] > fft_data[i + 1] && fft_data[i] >= minimum_height
            push!(peaks, i)
        end
    end

    if isempty(peaks)
        return 0.0, 0.0, 0.0
    end

    peak1_amp = maximum(fft_data[peaks])
    if prominence === nothing
        prominence = peak1_amp == 0 ? minimum_height : peak1_amp / 7
    end

    # Approximate SciPy prominence filtering with a height threshold.
    peaks = [i for i in peaks if fft_data[i] >= prominence]
    if isempty(peaks)
        return 0.0, 0.0, 0.0
    end

    sorted_peaks = sort(peaks, by=i -> fft_data[i], rev=true)
    first_idx = sorted_peaks[1]
    first_amp = fft_data[first_idx]
    first_freq = fft_freq_data[first_idx]
    second_freq = length(sorted_peaks) > 1 ? fft_freq_data[sorted_peaks[2]] : 0.0

    println("First harmonic: $first_amp at $first_freq")
    println("Second harmonic frequency: $second_freq")

    return first_freq, first_amp, second_freq
end

function trapz(y, x)
    n = length(y)
    if n < 2
        return 0.0
    end
    area = 0.0
    for i in 1:(n - 1)
        area += (x[i + 1] - x[i]) * (y[i + 1] + y[i]) / 2
    end
    return area
end

function integrate_first_harmonic(fft_data, freq_vals, first_harmonic_freq, second_harmonic_freq; harmonic_fraction=1.0)
    first_idx = argmin(abs.(freq_vals .- first_harmonic_freq))
    second_idx = argmin(abs.(freq_vals .- second_harmonic_freq))

    freq_diff = abs(freq_vals[second_idx] - freq_vals[first_idx])
    right_width = freq_diff * harmonic_fraction
    left_width = first_harmonic_freq * harmonic_fraction * 0.9

    lower_limit = first_harmonic_freq - left_width / 2
    upper_limit = first_harmonic_freq + right_width / 2

    lower_idx = argmin(abs.(freq_vals .- lower_limit))
    upper_idx = argmin(abs.(freq_vals .- upper_limit))

    area = trapz(fft_data[lower_idx:upper_idx], freq_vals[lower_idx:upper_idx]) * 2
    return area
end

function integrate_higher_modes(fft_data, freq_vals, first_harmonic_freq; harmonic_fraction=1.0)
    first_idx = argmin(abs.(freq_vals .- first_harmonic_freq))
    second_harmonic_freq = 2 * first_harmonic_freq
    second_idx = argmin(abs.(freq_vals .- second_harmonic_freq))

    freq_diff = freq_vals[second_idx] - freq_vals[first_idx]
    integration_width = freq_diff * harmonic_fraction
    upper_limit = first_harmonic_freq + integration_width / 2
    upper_idx = searchsortedlast(freq_vals, upper_limit)

    area = trapz(fft_data[(upper_idx + 1):end], freq_vals[(upper_idx + 1):end]) * 2
    return area
end

function analyse_fourier_data(sorted_files, freq_vals, norm_factor, is_temporal_ft; cut_index0=nothing, time_index_strategy="center_max")
    if is_temporal_ft && cut_index0 === nothing
        error("You must specify a cut index for temporal Fourier transform data")
    end

    first_mode_ft_peaks_amp = Float64[]
    first_mode_ft_peaks_freq = Float64[]
    first_mode_ft_peak_area = Float64[]
    higher_modes_ft_peak_area = Float64[]

    for file in sorted_files
        data = readdlm(file, Float64)
        if is_temporal_ft
            temporal_cut = find_temporal_cut_of_x_peaks(data, cut_index0)
            t_vals = data[:, 1]
            psi_cut_vals = temporal_cut
            freq_vals_full = fftfreq(length(t_vals), t_vals[2] - t_vals[1])
        else
            if time_index_strategy == "center_max"
                center_col = Int(cld(size(data, 2), 2))
                max_index = argmax(data[:, center_col])
            elseif time_index_strategy == "tracked_peak_max"
                temporal_cut = find_temporal_cut_of_x_peaks(data, cut_index0)
                max_index = argmax(temporal_cut)
            else
                error("Unknown time_index_strategy '$time_index_strategy'.")
            end
            psi_cut_vals = data[max_index, 2:end]
            freq_vals_full = freq_vals
        end

        half_len = floor(Int, length(psi_cut_vals) / 2)
        freq_vals_half = abs.(freq_vals_full[1:half_len])
        fft_vals = abs.(fft(psi_cut_vals) ./ length(psi_cut_vals))
        fft_vals = fft_vals[1:half_len] ./ norm_factor

        first_freq, first_amp, second_freq = find_first_harmonic(fft_vals, freq_vals_half)
        push!(first_mode_ft_peaks_amp, first_amp)
        push!(first_mode_ft_peaks_freq, first_freq)

        first_area = integrate_first_harmonic(fft_vals, freq_vals_half, first_freq, second_freq)
        push!(first_mode_ft_peak_area, first_area)

        higher_area = integrate_higher_modes(fft_vals, freq_vals_half, first_freq)
        push!(higher_modes_ft_peak_area, higher_area)
    end

    return Dict(
        "first_mode_ft_peaks_amp" => first_mode_ft_peaks_amp,
        "first_mode_ft_peaks_freq" => first_mode_ft_peaks_freq,
        "first_mode_ft_peak_area" => first_mode_ft_peak_area,
        "higher_modes_ft_peak_area" => higher_modes_ft_peak_area,
    )
end

function calc_spatial_mod_depth_vs_p0(sorted_files, nodes, x_index0, freq_vals)
    if isempty(sorted_files)
        error("No psi files found in the specified output directory.")
    end

    mod_depth_vals = Float64[]
    for file in sorted_files
        psi_data = readdlm(file, Float64)
        temporal_cut = find_temporal_cut_of_x_peaks(psi_data, x_index0)

        peak_time_index = argmax(temporal_cut)
        spatial_cut = psi_data[peak_time_index, 2:end]

        println("The max peak occurs at t = $(psi_data[peak_time_index, 1])")

        max_val = maximum(spatial_cut)
        min_val = minimum(spatial_cut)
        denom = max_val + min_val
        mod_depth = denom != 0 ? (max_val - min_val) / denom : 0.0
        push!(mod_depth_vals, mod_depth)
    end

    p0_vals = find_p0_vals_from_filenames(sorted_files)
    ft_data = analyse_fourier_data(sorted_files, freq_vals, 1.0, false; cut_index0=x_index0, time_index_strategy="tracked_peak_max")
    first_mode_amp = Float64.(ft_data["first_mode_ft_peaks_amp"])

    return p0_vals, mod_depth_vals, first_mode_amp
end

function create_plot(p0_vals, mod_depth_vals, ft_amp_vals, p0_samples, m_max_vals, p_th)
    fig, ax = subplots()

    # ax.plot(
    #     p0_vals,
    #     mod_depth_vals,
    #     color="tab:red",
    #     marker="x",
    #     linewidth=1,
    #     markersize=5,
    #     alpha=0.8,
    #     #label="Modulation depth",
    #     zorder=3,
    # )

    if !isempty(ft_amp_vals)
        ax.plot(
            p0_vals,
            ft_amp_vals,
            color="tab:red",
            marker="x",
            markersize=4,
            linewidth=1,
            alpha=0.8,
            #label="First harmonic amplitude",
            zorder=4,
        )
    end

    if !isempty(m_max_vals)
        ax.plot(
            p0_samples,
            m_max_vals,
            color="black",
            linewidth=1,
            #label="Analytic \$n_{q_c}(t = \bar{t_0})\$",
            zorder=5,
        )
    end

    ax.axvline(p_th, color="k", linestyle="--", linewidth=1.2,
    #label="\$p_{th}\$",
    zorder=2)

    ax.set_xlabel("\$p_0\$", fontsize=18)
    ax.set_ylabel(raw"$|n_{q_c}|_{\mathrm{max}}$", fontsize=18)
    ax.legend(frameon=false, loc="lower right", fontsize=16)
    # Force scientific notation with math text so ticks use ×10^n style.
    ax.ticklabel_format(axis="X", style="sci", scilimits=(0, 0), useMathText=true)

    if !isempty(p0_vals)
        ax.set_xlim(minimum(p0_vals) * 0.98, maximum(p0_vals) * 1.02)
    end
    ax.set_ylim(bottom=0)

    tight_layout()
    return fig, ax
end

function export_plot_data_csv(csv_path, p0_vals, ft_amp_vals, p0_samples, m_max_vals)
    n_rows = max(length(p0_vals), length(p0_samples))
    open(csv_path, "w") do io
        println(io, "p0_observed,first_harmonic_amp,p0_analytic,m_max_analytic")
        for i in 1:n_rows
            obs_p0 = i <= length(p0_vals) ? @sprintf("%.12e", p0_vals[i]) : ""
            obs_amp = i <= length(ft_amp_vals) ? @sprintf("%.12e", ft_amp_vals[i]) : ""
            ana_p0 = i <= length(p0_samples) ? @sprintf("%.12e", p0_samples[i]) : ""
            ana_m = i <= length(m_max_vals) ? @sprintf("%.12e", m_max_vals[i]) : ""
            println(io, "$(obs_p0),$(obs_amp),$(ana_p0),$(ana_m)")
        end
    end
end

function main()
    PyPlot.matplotlib["rcParams"]["ps.usedistiller"] = "xpdf"

    filename, xpos, csv_path = parse_args()

    output_dir = joinpath("patt1d_outputs", filename)
    input_dir = joinpath("patt1d_inputs", filename)
    seed_path = joinpath(input_dir, "seed.in")

    nodes, maxt, ht, width_psi, p0, Delta, gambar, b0, num_crit, R, gbar, v0, plotnum = read_input(seed_path)

    frac = (abs(xpos + pi * num_crit) / (2 * pi * num_crit)) * nodes
    x_index0 = clamp(Int(floor(frac)), 0, nodes - 1)

    x_vals = range(-pi * num_crit, pi * num_crit, length=nodes)
    dx = length(x_vals) > 1 ? x_vals[2] - x_vals[1] : 1.0
    k_vals = fftfreq(length(x_vals), dx)

    sorted_files = load_data(output_dir)
    p0_vals, mod_depth_vals, ft_amp_vals = calc_spatial_mod_depth_vs_p0(sorted_files, nodes, x_index0, k_vals)

    p_th = (2 * gambar) / (b0 * R)

    if !isempty(p0_vals)
        p0_min, p0_max = minimum(p0_vals), maximum(p0_vals)
        if p0_min == p0_max
            p0_samples = [p0_min]
        else
            p0_samples = range(p0_min, p0_max, length=max(length(p0_vals), 5000))
        end
        m_max_vals = sqrt.(2 * p_th .* max.(p0_samples .- p_th, 0) ./ (p0_samples .^ 2))
        m_max_vals = [p0_samples[i] > 0 ? m_max_vals[i] : NaN for i in eachindex(p0_samples)]
    else
        p0_samples = Float64[]
        m_max_vals = Float64[]
    end

    create_plot(p0_vals, mod_depth_vals, ft_amp_vals, collect(p0_samples), m_max_vals, p_th)
    if csv_path !== nothing
        export_plot_data_csv(csv_path, p0_vals, ft_amp_vals, collect(p0_samples), m_max_vals)
    end

    savefig("spatial_mod_depth_vs_p0.pdf", dpi=300)
    savefig("spatial_mod_depth_vs_p0.png", dpi=300)
    show()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
