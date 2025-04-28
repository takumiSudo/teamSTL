import torch
import numpy as np
from stlcgpp.formula import *
from stlcgpp.formulacvx import AlwaysCPG
import time


def test_always(signal, interval, verbose=True, **kwargs):
    def true_robustness_trace(signal, interval, **kwargs):
        T = len(signal)
        if interval is None:
            return torch.stack(
                [
                    minish(signal[i:], dim=0, keepdim=False, **kwargs)
                    for i in range(len(signal))
                ]
            )
        else:
            if interval[1] == torch.inf:
                large_number = 1e9
                start = interval[0]
                signal_padded = torch.cat(
                    [
                        signal,
                        torch.ones(T - 1) * large_number,
                        torch.ones(start) * -large_number,
                    ]
                )
                return torch.stack(
                    [
                        minish(
                            signal_padded[i + start : i + start + T],
                            dim=0,
                            keepdim=False,
                            **kwargs,
                        )
                        for i in range(len(signal))
                    ]
                )
            else:
                large_number = 1e9
                signal_padded = torch.cat(
                    [signal, torch.ones(interval[1] + 1) * -large_number]
                )
                return torch.stack(
                    [
                        minish(
                            signal_padded[i:][interval[0] : interval[1] + 1],
                            dim=0,
                            keepdim=False,
                            **kwargs,
                        )
                        for i in range(len(signal))
                    ]
                )

    def true_robustness(signal, interval, **kwargs):
        return true_robustness_trace(signal, interval, **kwargs)[0]

    pred = Predicate("identity", lambda x: x)
    phi = Always(pred > 0.0, interval=interval)
    rob = phi.robustness(signal, **kwargs)
    rob_trace = phi(signal, **kwargs)
    rob_grad = torch.func.grad(phi.robustness)(signal, **kwargs)

    true_trace = true_robustness_trace(signal, interval, **kwargs)
    true_rob = true_robustness(signal, interval, **kwargs)
    true_grad = torch.func.grad(true_robustness)(signal, interval, **kwargs)

    rob_correct = torch.isclose(rob, true_rob)
    trace_correct = torch.all(torch.isclose(rob_trace, true_trace))
    grad_correct = torch.all(torch.isclose(rob_grad, true_grad))

    pass_n = 0

    if rob_correct:
        if verbose:
            print("\u2713 Always robustness value match expected answer")
        pass_n += 1
    else:
        print("\u274c Always robustness value does not match expected answer")

    if trace_correct:
        if verbose:
            print("\u2713 Always robustness trace match expected answer")
        pass_n += 1
    else:
        print("\u274c Always robustness trace does not match expected answer")

    if grad_correct:
        if verbose:
            print("\u2713 Always robustness gradient match expected answer")
        pass_n += 1
    else:
        print("\u274c Always robustness gradient does not match expected answer")

    print("%i/3 test passed for Always formula\n" % pass_n)


def test_eventually(signal, interval, verbose=True, **kwargs):
    def true_robustness_trace(signal, interval, **kwargs):
        T = len(signal)
        if interval is None:
            return torch.stack(
                [
                    maxish(signal[i:], dim=0, keepdim=False, **kwargs)
                    for i in range(len(signal))
                ]
            )
        else:
            if interval[1] == torch.inf:
                large_number = 1e9
                start = interval[0]
                signal_padded = torch.cat([signal, torch.ones(T) * -large_number])
                return torch.stack(
                    [
                        maxish(
                            signal_padded[i + start : i + T],
                            dim=0,
                            keepdim=False,
                            **kwargs,
                        )
                        for i in range(len(signal))
                    ]
                )
            else:
                large_number = 1e9
                signal_padded = torch.cat(
                    [signal, torch.ones(interval[1] + 1) * -large_number]
                )
                return torch.stack(
                    [
                        maxish(
                            signal_padded[i:][interval[0] : interval[1] + 1],
                            dim=0,
                            keepdim=False,
                            **kwargs,
                        )
                        for i in range(len(signal))
                    ]
                )

    def true_robustness(signal, interval, **kwargs):
        return true_robustness_trace(signal, interval, **kwargs)[0]

    pred = Predicate("identity", lambda x: x)
    phi = Eventually(pred > 0.0, interval=interval)
    rob = phi.robustness(signal, **kwargs)
    rob_trace = phi(signal, **kwargs)
    rob_grad = torch.func.grad(phi.robustness)(signal, **kwargs)

    true_trace = true_robustness_trace(signal, interval, **kwargs)
    true_rob = true_robustness(signal, interval, **kwargs)
    true_grad = torch.func.grad(true_robustness)(signal, interval, **kwargs)

    rob_correct = torch.isclose(rob, true_rob)
    trace_correct = torch.all(torch.isclose(rob_trace, true_trace))
    grad_correct = torch.all(torch.isclose(rob_grad, true_grad))

    pass_n = 0

    if rob_correct:
        if verbose:
            print("\u2713 Eventually robustness value match expected answer")
        pass_n += 1
    else:
        print("\u274c Eventually robustness value does not match expected answer")

    if trace_correct:
        if verbose:
            print("\u2713 Eventually robustness trace match expected answer")
        pass_n += 1
    else:
        print("\u274c Eventually robustness trace does not match expected answer")

    if grad_correct:
        if verbose:
            print("\u2713 Eventually robustness gradient match expected answer")
        pass_n += 1
    else:
        print("\u274c Eventually robustness gradient does not match expected answer")

    print("%i/3 test passed for Eventually formula\n" % pass_n)


def test_until(signal, interval, verbose=True, **kwargs):
    pred = Predicate("identity", lambda x: x)
    phi = Until(pred > 0.0, pred < 0.0, interval=interval)

    def true_robustness_trace(signal, interval, **kwargs):
        signal1, signal2 = phi.subformula1(signal), phi.subformula2(signal)
        T = len(signal1)
        if interval is None:
            interval = [0, T - 1]
        elif interval[1] == torch.inf:
            interval = [interval[0], T - 1]
        large_number = 1e9
        signal1_padded = torch.cat([signal1, torch.ones_like(signal1) * -large_number])
        signal2_padded = torch.cat([signal2, torch.ones_like(signal2) * -large_number])
        return torch.stack(
            [
                maxish(
                    torch.stack(
                        [
                            minish(
                                torch.stack(
                                    [
                                        minish(
                                            signal1_padded[i:][: t + 1],
                                            dim=0,
                                            keepdim=False,
                                            **kwargs,
                                        ),
                                        signal2_padded[i:][t],
                                    ]
                                ),
                                dim=0,
                                keepdim=False,
                                **kwargs,
                            )
                            for t in range(interval[0], interval[-1] + 1)
                        ]
                    ),
                    dim=0,
                    keepdim=False,
                    **kwargs,
                )
                for i in range(T)
            ]
        )

    def true_robustness(signal, interval, **kwargs):
        return true_robustness_trace(signal, interval, **kwargs)[0]

    pred = Predicate("identity", lambda x: x)
    phi = Until(pred > 0.0, pred < 0, interval=interval)
    rob = phi.robustness(signal, **kwargs)
    rob_trace = phi(signal, **kwargs)
    rob_grad = torch.func.grad(phi.robustness)(signal, **kwargs)

    true_trace = true_robustness_trace(signal, interval, **kwargs)
    true_rob = true_robustness(signal, interval, **kwargs)
    true_grad = torch.func.grad(true_robustness)(signal, interval, **kwargs)

    rob_correct = torch.isclose(rob, true_rob, atol=1e-5)
    trace_correct = torch.all(torch.isclose(rob_trace, true_trace, atol=1e-5))
    grad_diff = torch.linalg.norm(rob_grad - true_grad)
    grad_correct = torch.all(torch.isclose(rob_grad, true_grad, atol=1e-5))

    pass_n = 0

    if rob_correct:
        if verbose:
            print("\u2713 Until robustness value match expected answer")
        pass_n += 1
    else:
        print("\u274c Until robustness value does not match expected answer")

    if trace_correct:
        if verbose:
            print("\u2713 Until robustness trace match expected answer")
        pass_n += 1
    else:
        print("\u274c Until robustness trace does not match expected answer")

    if grad_correct:
        if verbose:
            print("\u2713 Until robustness gradient match expected answer")
        pass_n += 1
    else:
        print("\u274c Until robustness gradient does not match expected answer")
        print(grad_diff)
        print(rob_grad, true_grad)

    print("%i/3 test passed for Until formula\n" % pass_n)


def test_always_mask_recurrent(signal, interval, verbose=True, **kwargs):
    signal_flip = signal.flip(0)
    pred = Predicate("identity", lambda x: x)
    mask = Always(pred > 0.0, interval)
    rec = AlwaysRecurrent(pred > 0.0, interval)

    rob_correct = torch.isclose(
        mask.robustness(signal, **kwargs), rec.robustness(signal_flip, **kwargs)
    )
    trace_correct = torch.all(
        torch.isclose(mask(signal, **kwargs), rec(signal_flip, **kwargs).flip(0))
    )
    grad_correct = torch.all(
        torch.isclose(
            torch.func.grad(mask.robustness)(signal, **kwargs),
            torch.func.grad(rec.robustness)(signal_flip, **kwargs).flip(0),
        )
    )

    pass_n = 0

    if rob_correct:
        if verbose:
            print(
                "\u2713 Always mask vs recurrent robustness value match expected answer"
            )
        pass_n += 1
    else:
        print(
            "\u274c Always mask vs recurrent robustness value does not match expected answer"
        )

    if trace_correct:
        if verbose:
            print(
                "\u2713 Always mask vs recurrent robustness trace match expected answer"
            )
        pass_n += 1
    else:
        print(
            "\u274c Always mask vs recurrent robustness trace does not match expected answer"
        )

    if grad_correct:
        if verbose:
            print(
                "\u2713 Always mask vs recurrent robustness gradient match expected answer"
            )
        pass_n += 1
    else:
        print(
            "\u274c Always mask vs recurrent robustness gradient does not match expected answer"
        )

    print("%i/3 test passed for Always mask vs recurrent formula\n" % pass_n)


# --- New test: AlwaysCPG vs Mask ---
def test_always_cpg_vs_mask(signal, interval, verbose=True, **kwargs):
    """
    Compare STLCG++ masking-based Always against AlwaysCPG LP-based Always.
    """
    pred = Predicate("identity", lambda x: x)
    phi_mask = Always(pred > 0.0, interval=interval)
    phi_cpg  = AlwaysCPG(pred > 0.0, interval=interval)

    # Evaluate robustness values
    rob_mask  = phi_mask.robustness(signal, **kwargs)
    rob_cpg   = phi_cpg.robustness(signal, **kwargs)

    # Evaluate full robustness traces
    trace_mask = phi_mask(signal, **kwargs)
    trace_cpg  = phi_cpg(signal, **kwargs)

    # Evaluate gradients
    grad_mask = torch.func.grad(phi_mask.robustness)(signal, **kwargs)
    grad_cpg  = torch.func.grad(phi_cpg.robustness)(signal, **kwargs)

    # Compare
    rob_correct   = torch.isclose(rob_mask, rob_cpg)
    trace_correct = torch.all(torch.isclose(trace_mask, trace_cpg))
    grad_correct  = torch.all(torch.isclose(grad_mask, grad_cpg))

    pass_n = 0
    if rob_correct:
        if verbose:
            print("✓ CPG vs Mask robustness value match")
        pass_n += 1
    else:
        print("✗ CPG vs Mask robustness value mismatch")

    if trace_correct:
        if verbose:
            print("✓ CPG vs Mask robustness trace match")
        pass_n += 1
    else:
        print("✗ CPG vs Mask robustness trace mismatch")

    if grad_correct:
        if verbose:
            print("✓ CPG vs Mask robustness gradient match")
        pass_n += 1
    else:
        print("✗ CPG vs Mask robustness gradient mismatch")

    print(f"{pass_n}/3 tests passed for AlwaysCPG vs Always\n")


# --- Benchmarking function: Always (mask) vs AlwaysCPG ---
def benchmark_always_methods(signal, interval, n_iters=50, **kwargs):
    """
    Benchmark average execution time of Always (mask) vs AlwaysCPG (CPG solver) for robustness().
    """
    pred = Predicate("identity", lambda x: x)
    phi_mask = Always(pred > 0.0, interval=interval)
    phi_cpg  = AlwaysCPG(pred > 0.0, interval=interval)

    # Warm-up runs
    phi_mask.robustness(signal, **kwargs)
    phi_cpg.robustness(signal, **kwargs)

    # Benchmark mask-based Always
    start = time.perf_counter()
    for _ in range(n_iters):
        phi_mask.robustness(signal, **kwargs)
    mask_time = (time.perf_counter() - start) / n_iters

    # Benchmark CPG-based Always
    start = time.perf_counter()
    for _ in range(n_iters):
        phi_cpg.robustness(signal, **kwargs)
    cpg_time = (time.perf_counter() - start) / n_iters

    print(f"Benchmark Always (mask) avg: {mask_time*1e3:.3f} ms/iter; "
          f"AlwaysCPG avg: {cpg_time*1e3:.3f} ms/iter "
          f"for interval={interval}")


def test_eventually_mask_recurrent(signal, interval, verbose=True, **kwargs):
    signal_flip = signal.flip(0)
    pred = Predicate("identity", lambda x: x)
    mask = Eventually(pred > 0.0, interval)
    rec = EventuallyRecurrent(pred > 0.0, interval)

    rob_correct = torch.isclose(
        mask.robustness(signal, **kwargs), rec.robustness(signal_flip, **kwargs)
    )
    trace_correct = torch.all(
        torch.isclose(mask(signal, **kwargs), rec(signal_flip, **kwargs).flip(0))
    )
    grad_correct = torch.all(
        torch.isclose(
            torch.func.grad(mask.robustness)(signal, **kwargs),
            torch.func.grad(rec.robustness)(signal_flip, **kwargs).flip(0),
        )
    )

    pass_n = 0

    if rob_correct:
        if verbose:
            print(
                "\u2713 Eventually mask vs recurrent robustness value match expected answer"
            )
        pass_n += 1
    else:
        print(
            "\u274c Eventually mask vs recurrent robustness value does not match expected answer"
        )

    if trace_correct:
        if verbose:
            print(
                "\u2713 Eventually mask vs recurrent robustness trace match expected answer"
            )
        pass_n += 1
    else:
        print(
            "\u274c Eventually mask vs recurrent robustness trace does not match expected answer"
        )

    if grad_correct:
        if verbose:
            print(
                "\u2713 Eventually mask vs recurrent robustness gradient match expected answer"
            )
        pass_n += 1
    else:
        print(
            "\u274c Eventually mask vs recurrent robustness gradient does not match expected answer"
        )

    print("%i/3 test passed for Eventually mask vs recurrent formula\n" % pass_n)


def test_until_mask_recurrent(signal, interval, verbose=True, **kwargs):
    signal_flip = signal.flip(0)
    pred = Predicate("identity", lambda x: x)
    mask = Until(pred > 0.0, pred < 0.0, interval)
    rec = UntilRecurrent(pred > 0.0, pred < 0.0, interval)

    rob_correct = torch.isclose(
        mask.robustness(signal, **kwargs), rec.robustness(signal_flip, **kwargs)
    )
    trace_correct = torch.all(
        torch.isclose(mask(signal, **kwargs), rec(signal_flip, **kwargs).flip(0))
    )
    grad_correct = torch.all(
        torch.isclose(
            torch.func.grad(mask.robustness)(signal, **kwargs),
            torch.func.grad(rec.robustness)(signal_flip, **kwargs).flip(0),
        )
    )

    pass_n = 0

    if rob_correct:
        if verbose:
            print(
                "\u2713 Until mask vs recurrent robustness value match expected answer"
            )
        pass_n += 1
    else:
        print(
            "\u274c Until mask vs recurrent robustness value does not match expected answer"
        )

    if trace_correct:
        if verbose:
            print(
                "\u2713 Until mask vs recurrent robustness trace match expected answer"
            )
        pass_n += 1
    else:
        print(
            "\u274c Until mask vs recurrent robustness trace does not match expected answer"
        )

    if grad_correct:
        if verbose:
            print(
                "\u2713 Until mask vs recurrent robustness gradient match expected answer"
            )
        pass_n += 1
    else:
        print(
            "\u274c Until mask vs recurrent robustness gradient does not match expected answer"
        )

    print("%i/3 test passed for Until mask vs recurrent formula\n" % pass_n)


def test_all_settings(test_func, verbose=True, T=10):
    assert T > 0, "Pick a T larger than 10"
    signal = torch.tensor(np.random.randn(T)).float() * 1.0
    # None, [0,b], [a,b], [0, inf], [a, inf]
    interval_list = [
        None,
        [0, T // 3],
        [T // 4, T // 2],
        [0, torch.inf],
        [T // 4, torch.inf],
    ]
    approx_method_list = ["true", "logsumexp"]
    temperature_list = [1.0, 10.0, 20.0, 100.0]

    for interval in interval_list:
        for approx_method in approx_method_list:
            for temperature in temperature_list:
                kwargs = {"approx_method": approx_method, "temperature": temperature}
                print(f"int={interval}\t temp={temperature} \t approx={approx_method} ")
                test_func(signal, interval, verbose, **kwargs)


if __name__ == "__main__":

    # test_all_settings(test_always, verbose=False, T=10)
    # test_all_settings(test_eventually, verbose=False, T=10)
    # test_all_settings(test_until, verbose=False, T=10)
    # test_all_settings(test_always_mask_recurrent, verbose=False, T=10)
    # test_all_settings(test_eventually_mask_recurrent, verbose=False, T=10)
    # test_all_settings(test_until_mask_recurrent, verbose=False, T=10)

    test_all_settings(test_always_cpg_vs_mask, verbose=False, T=10)

    # Benchmark performance of mask vs CPG
    print("\n--- Benchmarking Always robustness implementations ---")
    signal = torch.tensor(np.random.randn(20)).float()
    for interval in [None, [0, 5], [5, 15], [0, torch.inf], [10, torch.inf]]:
        benchmark_always_methods(signal, interval, n_iters=100, approx_method="true", temperature=1.0)
