from dataclasses import dataclass, field
import re
import polars as pl


@dataclass
class MatpowerCase:
    """Parsed MATPOWER case."""
    base_power_MVA: float | None = None
    tables: dict[str, pl.DataFrame] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    base_frequency_Hz:float = 60.0

    @classmethod
    def from_file(cls, filepath) -> 'MatpowerCase':
        """
        Parse a MATPOWER case file.

        Parameters
        ----------
        filepath
            Path to a MATPOWER ``.m`` case file.

        Returns
        -------
        MatpowerCase
            Parsed numerical tables and case metadata.

        Notes
        -----
        MATPOWER case format:
        https://matpower.org/docs/ref/matpower5.0/caseformat.html
        """

        with open(filepath, "r") as f:
            lines = f.readlines()

        case = MatpowerCase()

        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # ===============================================================
            # Scalar assignment
            #
            #     mpc.baseMVA = 100;
            #     mpc.version = '2';
            # ===============================================================
            m = re.match(r"mpc\.(\w+)\s*=\s*([^;]+);", line)

            if m:
                name, value = m.groups()

                value = value.strip()

                # String
                if (
                    len(value) >= 2
                    and value[0] in "'\""
                    and value[-1] == value[0]
                ):
                    value = value[1:-1]

                # Number
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass

                if name == "baseMVA":
                    case.base_power_MVA = float(value)
                else:
                    case.metadata[name] = value

                i += 1
                continue

            # ===============================================================
            # Numeric matrix
            #
            #     mpc.bus = [
            #         ...
            #     ];
            # ===============================================================
            m = re.match(r"mpc\.(\w+)\s*=\s*\[", line)

            if m:
                table_name = m.group(1)

                # Get column names from preceding comment.
                cols = []
                if i > 0:
                    comment = lines[i - 1].strip()
                    if comment.startswith("%"):
                        cols = comment[1:].split()

                rows = []

                i += 1
                while i < len(lines):
                    row_line = lines[i].strip()

                    if row_line.startswith("];"):
                        break

                    if row_line and not row_line.startswith("%"):
                        row_line = row_line.rstrip(";")
                        rows.append([
                            float(x)
                            for x in row_line.split()
                        ])

                    i += 1

                case.tables[table_name] = pl.from_records(
                    rows,
                    schema=cols if cols else None,
                )

                i += 1
                continue            

            i += 1

        return case

    def bus_data(self):
        """
        MATPOWER Schema
        ---------------
        Bus Data Format
            1   bus number (positive integer)
            2   bus type
                    PQ bus          = 1
                    PV bus          = 2
                    reference bus   = 3
                    isolated bus    = 4
            3   Pd, real power demand (MW)
            4   Qd, reactive power demand (MVAr)
            5   Gs, shunt conductance (MW demanded at V = 1.0 p.u.)
            6   Bs, shunt susceptance (MVAr injected at V = 1.0 p.u.)
            7   area number, (positive integer)
            8   Vm, voltage magnitude (p.u.)
            9   Va, voltage angle (degrees)
        (-)     (bus name)
            10  baseKV, base voltage (kV)
            11  zone, loss zone (positive integer)
        (+) 12  maxVm, maximum voltage magnitude (p.u.)
        (+) 13  minVm, minimum voltage magnitude (p.u.)
        """
        df = (
            self.tables["bus"]
            .rename({"baseKV":"base_voltage_kV", "Vmax":"maximum_voltage_pu", "Vmin":"minimum_voltage_pu"})
            .with_columns(
                name = pl.col("bus_i").cast(pl.Int64).cast(pl.String),
                bus_type = pl.col("type").cast(pl.String).replace({"1.0":"PQ", "2.0":"PV", "3.0":"slack", "4.0":"isolated"}),
                base_frequency_Hz = pl.lit(self.base_frequency_Hz),
                base_power_MVA = pl.lit(self.base_power_MVA),
            )
            .select("name", "bus_type", "base_power_MVA", "base_voltage_kV", "base_frequency_Hz", "minimum_voltage_pu", "maximum_voltage_pu")
        )
        return df

    def shunt_data(self):
        df = (
            self.tables["bus"]
            .rename({"Gs":"g_pu", "Bs":"b_pu", "baseKV":"base_voltage_kV"})
            .with_columns(
                bus = pl.col("bus_i").cast(pl.Int64).cast(pl.String),
                base_frequency_Hz = pl.lit(self.base_frequency_Hz),
                base_power_MVA = pl.lit(self.base_power_MVA),
            )
            # Drop buses without shunts
            .filter(
                (pl.col("b_pu") != 0) | (pl.col("g_pu") != 0)
            )
            .select("bus", "b_pu", "g_pu", "base_power_MVA", "base_voltage_kV", "base_frequency_Hz")
        )
        return df

    def load_data(self):
        df = (
            self.tables["bus"]
            .rename({"Pd":"load_MW", "Qd":"load_MVAR"})
            .with_columns(
                bus = pl.col("bus_i").cast(pl.Int64).cast(pl.String),
                timepoint = pl.lit("t0"),
                load_MVAR = pl.col("load_MVAR").abs()
            )
            .select("bus", "load_MW", "load_MVAR", "timepoint")
            # Drop buses without loads
            .filter(
               (pl.col("load_MW") != 0) |  (pl.col("load_MVAR") != 0)
            )
        )
        return df

    def line_data(self):
        """
        MATPOWER Schema
        ---------------
        Branch Data Format
            1   f, from bus number
            2   t, to bus number
        (-)     (circuit identifier)
            3   r, resistance (p.u.)
            4   x, reactance (p.u.)
            5   b, total line charging susceptance (p.u.)
            6   rateA, MVA rating A (long term rating)
            7   rateB, MVA rating B (short term rating)
            8   rateC, MVA rating C (emergency rating)
            9   ratio, transformer off nominal turns ratio ( = 0 for lines )
                (taps at 'from' bus, impedance at 'to' bus,
                    i.e. if r = x = 0, then ratio = Vf / Vt)
            10  angle, transformer phase shift angle (degrees), positive => delay
        (-)     (Gf, shunt conductance at from bus p.u.)
        (-)     (Bf, shunt susceptance at from bus p.u.)
        (-)     (Gt, shunt conductance at to bus p.u.)
        (-)     (Bt, shunt susceptance at to bus p.u.)
            11  initial branch status, 1 - in service, 0 - out of service
        (2) 12  minimum angle difference, angle(Vf) - angle(Vt) (degrees)
        (2) 13  maximum angle difference, angle(Vf) - angle(Vt) (degrees)
                (The voltage angle difference is taken to be unbounded below
                    if ANGMIN < -360 and unbounded above if ANGMAX > 360.
                    If both parameters are zero, it is unconstrained.)
        """
        # Median values per mile from IEEE RTS 79 (https://labs.ece.uw.edu/pstca/rts/pg_tcarts.htm)
        # Assuming 230 parameters = 345 parameters 
        r_pu_per_mile = {138:0.001, 230:0.000182, 345:0.000182}
        x_pu_per_mile = {138:0.003837, 230:0.001447, 345:0.001447}
        b_pu_per_mile = {138:0.001045, 230:0.00303, 345:0.00303}

        df = (
            self.tables["branch"]
            # Rename columns and process data types
            .rename({'r':'r_pu', 'x':'x_pu', 'b':'b_pu', 'angmin':'angle_min_deg', 'angmax':'angle_max_deg'})
            .with_columns(
                from_bus = pl.col("fbus").cast(pl.Int64).cast(pl.String),
                to_bus = pl.col("tbus").cast(pl.Int64).cast(pl.String),
                base_power_MVA = pl.lit(self.base_power_MVA),
                base_frequency_Hz = pl.lit(self.base_frequency_Hz)
            )
            # Filter out any "out of service" lines
            .filter(pl.col('status') == 1.0)
            # Inherit line voltage from the "from_bus". Assumes a "ratio" = 1 for all branches
            .join(
                self.tables["bus"].select("bus_i", "baseKV"),
                left_on='fbus',
                right_on="bus_i"
            )
            .rename({"baseKV":"base_voltage_kV"})
            # Estimate line parameters
            .with_columns(
                name = pl.col("from_bus")+"_to_" + pl.col("to_bus"),
                x_pu_mile = pl.col("base_voltage_kV").replace(x_pu_per_mile, default=None),
                r_pu_mile = pl.col("base_voltage_kV").replace(r_pu_per_mile, default=None),
                b_pu_mile = pl.col("base_voltage_kV").replace(b_pu_per_mile, default=None),
            )
            .with_columns(
                estimated_miles = pl.col("x_pu") / pl.col("x_pu_mile")
            )
            .with_columns(
                r_pu = pl.when(pl.col("r_pu") == 0).then(pl.col("r_pu_mile")*pl.col("estimated_miles")).otherwise(pl.col("r_pu")),
                b_pu = pl.when(pl.col("b_pu") == 0).then(pl.col("b_pu_mile")*pl.col("estimated_miles")).otherwise(pl.col("b_pu")),
            )
            .with_columns(
                g_pu = 0.01 * pl.col("b_pu")
            )
            # Select final columns
            .select("name", "from_bus", "to_bus", "r_pu", "x_pu", "g_pu", "b_pu", "base_power_MVA", "base_voltage_kV", "base_frequency_Hz")
        )

        return df

    def generator_data(self, force_dispatch_tol=None):
        """
        Parameters
        ----------
        force_dispatch_tol: float 
        - Force dispatch at MATPOWER level to within a tolerance, by default 
            do not force dispatch.

        MATPOWER Schema
        ---------------
        Generator Data Format
            1   bus number
        (-)     (machine identifier, 0-9, A-Z)
            2   Pg, real power output (MW)
            3   Qg, reactive power output (MVAr)
            4   Qmax, maximum reactive power output (MVAr)
            5   Qmin, minimum reactive power output (MVAr)
            6   Vg, voltage magnitude setpoint (p.u.)
        (-)     (remote controlled bus index)
            7   mBase, total MVA base of this machine, defaults to baseMVA
        (-)     (machine impedance, p.u. on mBase)
        (-)     (step up transformer impedance, p.u. on mBase)
        (-)     (step up transformer off nominal turns ratio)
            8   status,  >  0 - machine in service
                            <= 0 - machine out of service
        (-)     (% of total VAr's to come from this gen in order to hold V at
                    remote bus controlled by several generators)
            9   Pmax, maximum real power output (MW)
            10  Pmin, minimum real power output (MW)
        (2) 11  Pc1, lower real power output of PQ capability curve (MW)
        (2) 12  Pc2, upper real power output of PQ capability curve (MW)
        (2) 13  Qc1min, minimum reactive power output at Pc1 (MVAr)
        (2) 14  Qc1max, maximum reactive power output at Pc1 (MVAr)
        (2) 15  Qc2min, minimum reactive power output at Pc2 (MVAr)
        (2) 16  Qc2max, maximum reactive power output at Pc2 (MVAr)
        (2) 17  ramp rate for load following/AGC (MW/min)
        (2) 18  ramp rate for 10 minute reserves (MW)
        (2) 19  ramp rate for 30 minute reserves (MW)
        (2) 20  ramp rate for reactive power (2 sec timescale) (MVAr/min)
        (2) 21  APF, area participation factor
        """
        df = (
            self.tables["gen"]
            .join(
                self.tables["bus"].select("bus_i", "baseKV"),
                left_on='bus',
                right_on="bus_i"
            )
            .rename({
                "Qmax":"maximum_reactive_power_MVAR", 
                "Qmin":"minimum_reactive_power_MVAR", 
                "Pmax":"maximum_active_power_MW", 
                "Pmin":"minimum_active_power_MW",
                "mBase":"base_power_MVA",
                "baseKV":"base_voltage_kV"}
                )
            .with_columns(
                bus = pl.col("bus").cast(pl.Int64).cast(pl.String),
                base_frequency_Hz = pl.lit(self.base_frequency_Hz),
            )
        )

        if force_dispatch_tol:
            df = (
                df
                .with_columns(
                    maximum_reactive_power_MVAR = pl.col("Qg") + force_dispatch_tol,
                    minimum_reactive_power_MVAR= pl.col("Qg") - force_dispatch_tol,
                    maximum_active_power_MW= pl.col("Pg") + force_dispatch_tol,
                    minimum_active_power_MW= pl.col("Pg") - force_dispatch_tol,
                )
            )

        df = df.select("bus", "base_power_MVA", "minimum_active_power_MW", "maximum_active_power_MW", "minimum_reactive_power_MVAR", "maximum_reactive_power_MVAR", "base_frequency_Hz")

        return df

    def to_system(self, generators=None, force_dispatch_tol=None):
        from sting import main
        from sting.bus.core import Bus
        from sting.line.pi_model import LinePiModel
        from sting.load.core import Load
        from sting.shunt.parallel_rc_shunt_2a import ParallelRCShunt2A
        from sting.system.core import System
        from sting.timescales import Timepoint

        system = System()
        system.add(Timepoint(name="t0", weight=1))

        for row in self.bus_data().iter_rows(named=True):
            system.add(Bus(**row))

        for row in self.line_data().iter_rows(named=True):
            system.add(LinePiModel(**row))

        for row in self.load_data().iter_rows(named=True):
            system.add(Load(**row))
            

        for row in self.shunt_data().iter_rows(named=True):
            system.add(ParallelRCShunt2A(**row))

        if generators is not None:
            for gen, row in zip(generators, self.generator_data(force_dispatch_tol).iter_rows(named=True)):
                for attribute in row.keys():
                    # Transfer generator data from MATPOWER to each instance
                    setattr(gen, attribute, row[attribute])
                system.add(gen)

        system.apply("post_system_init", system)

        return system