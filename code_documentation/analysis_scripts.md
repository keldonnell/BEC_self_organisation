# Analysis Scripts (Quick Reference)

All analysis scripts are in [`analysis/`](../analysis). They read simulation data from:

- `patt1d_inputs/<filename>/seed.in`
- `patt1d_outputs/<filename>/...`

## How To Run

From project root:

```bash
python analysis/<script_name>.py -h
```

Typical usage pattern:

```bash
python analysis/<script_name>.py -f <sim_filename> [other flags]
```

## Script List

| Script | What it does | Typical run |
|---|---|---|
| `analyze_delay_times_1D_BEC.py` | Finds observed delay times from first-harmonic peaks and compares with analytic scaling. | `python analysis/analyze_delay_times_1D_BEC.py -f <sim_filename> -x 0.0` |
| `calc_area_of_ft_data.py` | Computes Fourier-peak areas over selected frame ranges. | `python analysis/calc_area_of_ft_data.py -f <sim_filename> -x 0 -s 0 -e 30 -i 20` |
| `calc_legett_at_t.py` | Computes Leggett-style quantity at a selected time/frame. | `python analysis/calc_legett_at_t.py -f <sim_filename> -t 1e8 -i 20` |
| `calc_oscill_period_at_t.py` | Estimates oscillation period at a selected time/frame. | `python analysis/calc_oscill_period_at_t.py -f <sim_filename> -t 1e8 -i 20` |
| `compare_sqrt_vs_mean.py` | Demonstrates difference between `<sqrt(S)>` and `sqrt(<S>)` on a toy profile. | `python analysis/compare_sqrt_vs_mean.py` |
| `export_s_statistics.py` | Exports spatial profiles of `<sqrt(abs(S))>` and `sqrt(<abs(S)>)` after analytic delay `t0`. | `python analysis/export_s_statistics.py -f <sim_filename> -i 25` |
| `plot_analytic_and_observed_omega_gamma.py` | Plots analytic vs observed frequency/growth diagnostics vs `p0`. | `python analysis/plot_analytic_and_observed_omega_gamma.py -f <sim_filename> -x 0` |
| `plot_density_evolution_at_t.py` | Plots density `|psi|^2(x)` at one time. | `python analysis/plot_density_evolution_at_t.py -f <sim_filename> -t 1e8 -i 20` |
| `plot_density_evolution_at_x.py` | Plots density `|psi|^2(t)` at one spatial position. | `python analysis/plot_density_evolution_at_x.py -f <sim_filename> -x 0 -i 20` |
| `plot_density_first_harmonic_max_vs_p0.py` | For each density file, finds max first-harmonic amplitude vs time and plots it vs `p0`. | `python analysis/plot_density_first_harmonic_max_vs_p0.py -f <sim_filename>` |
| `plot_first_harmonic_amp_vs_time_density.py` | Plots density first-harmonic amplitude vs time for one frame index. | `python analysis/plot_first_harmonic_amp_vs_time_density.py -f <sim_filename> -i 20` |
| `plot_first_harmonic_amp_vs_time_intensity.py` | Plots intensity first-harmonic amplitude vs time for one frame index. | `python analysis/plot_first_harmonic_amp_vs_time_intensity.py -f <sim_filename> -i 20` |
| `plot_ft_at_t.py` | Plots spatial Fourier transform at a selected time. | `python analysis/plot_ft_at_t.py -f <sim_filename> -t 1e8 -i 20` |
| `plot_ft_at_x.py` | Plots temporal Fourier transform at a selected spatial position. | `python analysis/plot_ft_at_x.py -f <sim_filename> -x 0 -i 20` |
| `plot_intensity_first_harmonic_max_vs_p0.py` | Intensity analogue of max first-harmonic vs `p0`. | `python analysis/plot_intensity_first_harmonic_max_vs_p0.py -f <sim_filename>` |
| `plot_s_evolution_at_x.py` | Plots intensity `S(t)` at one spatial position. | `python analysis/plot_s_evolution_at_x.py -f <sim_filename> -x 0 -i 20` |
| `plot_s_ft_at_t.py` | Plots spatial Fourier transform of intensity at one time. | `python analysis/plot_s_ft_at_t.py -f <sim_filename> -i 20 -t 1e8` |
| `plot_spatial_analysis_graphs_changing_p0.py` | Multi-panel spatial statistics vs `p0` (density). | `python analysis/plot_spatial_analysis_graphs_changing_p0.py -f <sim_filename>` |
| `plot_spatial_ft_graphs_changing_p0.py` | Multi-panel spatial Fourier metrics vs `p0`. | `python analysis/plot_spatial_ft_graphs_changing_p0.py -f <sim_filename>` |
| `plot_spatial_mod_depth_vs_p0.py` | Spatial modulation depth / first-harmonic metrics vs `p0` (density). | `python analysis/plot_spatial_mod_depth_vs_p0.py -f <sim_filename> -x 0` |
| `plot_spatial_mod_depth_vs_p0_intensity.py` | Spatial first-harmonic metric vs `p0` (intensity). | `python analysis/plot_spatial_mod_depth_vs_p0_intensity.py -f <sim_filename> -x 0` |
| `plot_temporal_analysis_graphs_changing_p0_at_x.py` | Temporal statistics vs `p0` at fixed `x` (density). | `python analysis/plot_temporal_analysis_graphs_changing_p0_at_x.py -f <sim_filename> -x 0` |
| `plot_temporal_delay_vs_p0_at_x_using_fourier_amp.py` | Delay time vs `p0` from first-harmonic temporal peaks + analytic overlay. | `python analysis/plot_temporal_delay_vs_p0_at_x_using_fourier_amp.py -f <sim_filename> -x 0` |
| `plot_temporal_ft_graphs_changing_p0_at_x.py` | Multi-panel temporal Fourier metrics vs `p0` at fixed `x`. | `python analysis/plot_temporal_ft_graphs_changing_p0_at_x.py -f <sim_filename> -x 0` |
| `plot_temporal_mod_depth_vs_p0_at_x.py` | Temporal modulation depth vs `p0` at fixed `x` (density). | `python analysis/plot_temporal_mod_depth_vs_p0_at_x.py -f <sim_filename> -x 0` |
| `reindex_p0_outputs.py` | Renames `psi*.out` and `s*.out` so indices increase with `p0`. | `python analysis/reindex_p0_outputs.py -f <sim_filename> --dry-run` |

## Notes

- Use `-h` for full options on any script.
- Many scripts open interactive windows; some also save files.
- Most scripts assume one simulation folder name passed via `-f`.
