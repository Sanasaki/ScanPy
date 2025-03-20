import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


def infer_header(csvFilePath: str) -> list[str]:
    f = open(csvFilePath, "r")
    f.readline()
    header = f.readline().strip("\n").split(" ")
    f.close()
    return header


def graph(tableFile: str, yAxis: str, xAxis: str):
    # alias list
    # produce a nice png with convenient legend
    # lammpsThermo: list[str] = [
    #     "Step",
    #     "SimTime(ps)",
    #     "CpuTime(s)",
    #     "Temperature(K)",
    #     "Pressure(bar)",
    #     "Density(g/cm^3)",
    #     "Volume(A^3)",
    #     "KineticEnergy(kJ/mol.at)",
    #     "PotentialEnergy(kJ/mol.at)",
    #     "TotalEnergy(kJ/mol.at)",
    #     "Enthalpy(kJ/mol.at)",
    # ]
    lammpsThermo = infer_header(tableFile)
    print(lammpsThermo)
    lammpsDataFrame: pd.DataFrame = pd.read_csv(tableFile, skiprows=2, names=lammpsThermo, sep=" ", dtype=float)
    print(lammpsDataFrame)
    # lammpsDataFrame.plot(x=xAxis, y=yAxis)
    # plt.show()

    # add option to also infer some modelisation of the data
    return lammpsDataFrame


def drop_initialization_data(df: pd.DataFrame, initialization_time: float) -> pd.DataFrame:
    """
    Supprime les lignes où la deuxième colonne du DataFrame (le temps de simulation) est inférieure à initialization_time.
    """
    second_col_name = df.columns[1]  # Récupère le nom de la deuxième colonne
    return df[df[second_col_name] >= initialization_time].reset_index(drop=True)


def average_second_half(df: pd.DataFrame, npt_time: float) -> pd.DataFrame:
    """
    Calcule la moyenne de la température et de l'enthalpie sur la deuxième moitié de chaque intervalle NPT.
    """
    second_col_name = df.columns[1]  # Temps de simulation
    temp_col = "T(K)" if "T(K)" in df.columns else df.columns[3]  # Température
    enthalpy_col = "H(kJ/mol.at)" if "H(kJ/mol.at)" in df.columns else df.columns[-1]  # Enthalpie

    # Grouper par intervalle de temps NPT
    df["NPT_Group"] = (df[second_col_name] // npt_time).astype(int)

    # Sélectionner la deuxième moitié de chaque intervalle
    filtered_df = (
        df.groupby("NPT_Group", group_keys=False)
        .apply(lambda group: group[group[second_col_name] >= group[second_col_name].median()])
        .reset_index(drop=True)
    )

    # Moyenne des valeurs pour chaque groupe
    result = filtered_df.groupby("NPT_Group")[[temp_col, enthalpy_col]].mean().reset_index()

    return result


def plot_avg_enthalpy_vs_temperature(df_avg: pd.DataFrame, temp_col: str, enthalpy_col: str):
    """
    Trace un nuage de points de l'enthalpie moyenne en fonction de la température moyenne.
    """
    plt.scatter(df_avg[temp_col], df_avg[enthalpy_col], color="blue", alpha=0.7)
    plt.xlabel("Température (K)")
    plt.ylabel("Enthalpie (kJ/mol.at)")
    plt.title("Nuage de points : Enthalpie moyenne vs Température moyenne")
    plt.grid(True)
    plt.show()


def fit_best_polynomial(x, y, r2_threshold=0.95, max_degree=5):
    """
    Trouve le polynôme de meilleur degré pour approximer les données avec un R² supérieur au seuil.
    """
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)

    for degree in range(1, max_degree + 1):
        poly = PolynomialFeatures(degree)
        x_poly = poly.fit_transform(x)
        model = LinearRegression().fit(x_poly, y)
        y_pred = model.predict(x_poly)
        r2 = r2_score(y, y_pred)

        if r2 >= r2_threshold:
            coefficients = np.append(model.intercept_, model.coef_[1:])  # On exclut le premier coef qui correspond à x^0
            polynome = "y = "
            for i, coeff in enumerate(coefficients):
                polynome += f"{coeff:.4f}x^{i}"
                if i < len(coefficients) - 1:
                    polynome += " + "
            print(polynome)
            return model, poly, degree, r2

    return model, poly, max_degree, r2  # Retourne le modèle de degré max si aucun seuil n'est atteint


def graph_heat_capacity(lammpsDataPandas: pd.DataFrame, NPTdurationps: int, NVTdurationps: int):
    """
    Compute then graph the heat capacity as a function of the temperature.
    """
    # higher level function?
    thermoPostInit = drop_initialization_data(lammpsDataPandas, NVTdurationps)
    thermoEquilbriumData = average_second_half(thermoPostInit, NPTdurationps)
    plot_avg_enthalpy_vs_temperature(thermoEquilbriumData, "T(K)", "H(kJ/mol.at)")

    model, poly, degree, r2 = fit_best_polynomial(thermoEquilbriumData["T(K)"], thermoEquilbriumData["H(kJ/mol.at)"])
    print(f"Meilleur ajustement: Polynôme de degré {degree} avec R² = {r2:.4f}")
    # print(thermoEquilbriumData)
    # removing initialization
    # needs to take vibrations into account
    pass


if __name__ == "__main__":
    csvFile: str = r"C:\Users\JL252842\Documents\Thesis\GitHub\ScanPy\ScanPy\FixDataGlobal.csv"
    lammpsData = graph(
        csvFile,
        "Enthalpy(kJ/mol.at)",
        "Temperature(K)",
    )
    graph_heat_capacity(lammpsData, 20, 20)
