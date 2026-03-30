# Plot temporal delay time versus p0 at a fixed x position (Julia version)
# Matches the plotting style used in plot_spatial_mod_depth_vs_p0.jl

using PyPlot
using DelimitedFiles

const MIN_ABOVE_THRESHOLD_SCALING = 1e-3
const MIN_SEED_MAGNITUDE = 1e-30

function parse_args()
    filename = nothing
    xpos = nothing
    prominence_factor = 0.25

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "-f" || arg == "--filename"
            i += 1
            filename = ARGS[i]
        elseif arg == "-x" || arg == "--xpos"
            i += 1
            xpos = parse(Float64, ARGS[i])
        elseif arg == "--prominence-factor"
            i += 1
            prominence_factor = parse(Float64, ARGS[i])
        elseif arg == "-h" || arg == "--help"
            println("Usage: julia plot_temporal_delay_vs_p0_at_x.jl -f <input_dir_name> -x <x_position> [--prominence-factor <fraction>]")
            exit(0)
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end

    if filename === nothing || xpos === nothing
        error("Missing required arguments. Use -f <input_dir_name> and -x <x_position>.")
    end

    return filename, xpos, prominence_factor
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

    if length(values) < 14
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
        Int(values[13]),
        values[14],
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

function pump_threshold(gamma_bar, b0, reflectivity)
    if b0 == 0 || reflectivity == 0
        return Inf
    end
    return (2.0 * gamma_bar) / (b0 * reflectivity)
end

function analytic_delay_time(p0_vals::AbstractVector{<:Real}, p_th, gamma_bar, seed)
    results = fill(Inf, length(p0_vals))
    if !isfinite(p_th) || gamma_bar == 0 || abs(seed) < MIN_SEED_MAGNITUDE
        return results
    end

    for (i, p0) in enumerate(p0_vals)
        if !isfinite(p0) || p0 <= p_th
            continue
        end
        ratio = p0 / p_th
        excess = ratio - 1.0
        if excess <= 0
            continue
        end
        numerator = sqrt(2.0) * (p_th / p0) * sqrt(excess) / abs(seed)
        if numerator <= 1.0
            continue
        end
        denom = sqrt(excess) * abs(gamma_bar)
        results[i] = acosh(numerator) / denom
    end

    return results
end

function analytic_delay_time(p0::Real, p_th, gamma_bar, seed)
    return analytic_delay_time([p0], p_th, gamma_bar, seed)[1]
end

function find_first_peak_time(psi_data, x_index0, prominence_factor)
    spatial_data = psi_data[:, 2:end]
    col_index = clamp(x_index0 + 1, 1, size(spatial_data, 2))
    trace = spatial_data[:, col_index]

    max_val = maximum(trace)
    min_required = prominence_factor * max_val

    # Approximate SciPy's find_peaks prominence logic.
    function peak_prominence(data, idx)
        peak_val = data[idx]

        left_min = peak_val
        for j in (idx - 1):-1:1
            left_min = min(left_min, data[j])
            if data[j] >= peak_val
                break
            end
        end

        right_min = peak_val
        for j in (idx + 1):length(data)
            right_min = min(right_min, data[j])
            if data[j] >= peak_val
                break
            end
        end

        return peak_val - max(left_min, right_min)
    end

    i = 2
    n = length(trace)
    while i <= n - 1
        if trace[i] > trace[i - 1] && trace[i] > trace[i + 1]
            if peak_prominence(trace, i) >= min_required
                return psi_data[i, 1]
            end
        elseif trace[i] > trace[i - 1] && trace[i] == trace[i + 1]
            j = i + 1
            while j <= n && trace[j] == trace[i]
                j += 1
            end
            if j <= n && trace[j] < trace[i]
                plateau_idx = Int(floor((i + j - 1) / 2))
                if peak_prominence(trace, plateau_idx) >= min_required
                    return psi_data[plateau_idx, 1]
                end
            end
            i = j
            continue
        end
        i += 1
    end

    println("Didn't find any peaks")
    return NaN
end

function compute_delay_times(sorted_files, p0_vals, x_index0, prominence_factor)
    delay_times = Float64[]
    valid_p0 = Float64[]

    for (file, p0) in zip(sorted_files, p0_vals)
        data = readdlm(file, Float64)
        t0 = find_first_peak_time(data, x_index0, prominence_factor)
        if isnan(t0)
            continue
        end
        push!(valid_p0, p0)
        push!(delay_times, t0)
    end

    return valid_p0, delay_times
end

function find_vals_above_th(p0_vals, sorted_files, p_th)
    min_above_th = p_th * MIN_ABOVE_THRESHOLD_SCALING
    min_ind = findfirst(p0_vals .> (p_th + min_above_th))
    if min_ind === nothing
        return Float64[], String[]
    end
    println("p_th is $p_th")
    println("THE MIN IS $min_ind")
    return p0_vals[min_ind:end], sorted_files[min_ind:end]
end

function create_plot(p0_vals, delay_times, p0_analytic_vals, t0_analytic_vals, p_th)
    fig, ax = subplots()

    ax.plot(
        p0_analytic_vals,
        t0_analytic_vals,
        color="black",
        linewidth=1,
        zorder=2,
        #label="Analytic \$t_0\$",
    )

    ax.plot(
        p0_vals,
        delay_times,
        color="tab:red",
        marker="x",
        linestyle="None",
        markersize=5,
        linewidth=1,
        alpha=0.8,
        zorder=3,
        #label="Observed \$t_0\$",
    )

    ax.axvline(p_th, color="k", linestyle="--", linewidth=1.2, zorder=1)

    ax.set_xlabel("Pump strength \$p_0\$", fontsize=18)
    ax.set_ylabel("Delay time \$t_0\$", fontsize=18)
    ax.legend(frameon=false, loc="lower right", fontsize=16)

    if !isempty(p0_vals)
        ax.set_xlim(minimum(p0_vals) * 0.98, maximum(p0_vals) * 1.02)
    end
    ax.set_ylim(bottom=0)

    tight_layout()
    return fig, ax
end

function main()
    PyPlot.matplotlib["rcParams"]["ps.usedistiller"] = "xpdf"

    filename, xpos, prominence_factor = parse_args()

    output_dir = joinpath("patt1d_outputs", filename)
    input_dir = joinpath("patt1d_inputs", filename)
    seed_path = joinpath(input_dir, "seed.in")

    nodes, maxt, ht, width_psi, p0, Delta, gamma_bar, b0, num_crit, R, gbar, v0, plotnum, seed = read_input(seed_path)

    frac = (abs(xpos + pi * num_crit) / (2 * pi * num_crit)) * nodes
    x_index0 = clamp(Int(floor(frac)), 0, nodes - 1)

    sorted_files = load_data(output_dir)
    all_p0_vals = find_p0_vals_from_filenames(sorted_files)

    p_th = pump_threshold(gamma_bar, b0, R)
    p0_vals, sorted_files_above_th = find_vals_above_th(all_p0_vals, sorted_files, p_th)
    p0_vals, delay_times = compute_delay_times(sorted_files_above_th, p0_vals, x_index0, prominence_factor)

    if isempty(p0_vals)
        error("Unable to determine any delay times above threshold.")
    end

    above_threshold = p0_vals[p0_vals .> (p_th * 1.001)]
    p0_min_for_curve = isempty(above_threshold) ? minimum(p0_vals) : minimum(above_threshold)
    p0_analytic_vals = collect(range(p0_min_for_curve, maximum(p0_vals), length=2000))
    t0_analytic_vals = analytic_delay_time(p0_analytic_vals, p_th, gamma_bar, seed)

    create_plot(p0_vals, delay_times, p0_analytic_vals, t0_analytic_vals, p_th)

    savefig("temporal_delay_vs_p0_at_x.pdf", dpi=300)
    savefig("temporal_delay_vs_p0_at_x.png", dpi=300)
    show()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
