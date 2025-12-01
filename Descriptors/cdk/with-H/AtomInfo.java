import java.util.TreeMap;

public class AtomInfo {

    // 分子名
    String title = "";

    // index
    int index = -1;

    // symbole
    String symbol = "";

    // AtomType
    String atomType = "";

    // 記述子計算の結果を格納するマップ
    TreeMap<String, String> descriptorResultMap = new TreeMap<String, String>();

    int longestMaxTopDistInMolecule;

    int highestMaxTopDistInMatrixRow;

    public AtomInfo() {
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }

    public String getSymbol() {
        return symbol;
    }

    public void setSymbol(String symbol) {
        this.symbol = symbol;
    }

    public String getAtomType() {
        return atomType;
    }

    public void setAtomType(String atomType) {
        this.atomType = atomType;
    }

    public int getLongestMaxTopDistInMolecule() {
        return longestMaxTopDistInMolecule;
    }

    public void setLongestMaxTopDistInMolecule(int longestMaxTopDistInMolecule) {
        this.longestMaxTopDistInMolecule = longestMaxTopDistInMolecule;
    }

    public int getHighestMaxTopDistInMatrixRow() {
        return highestMaxTopDistInMatrixRow;
    }

    public void setHighestMaxTopDistInMatrixRow(int highestMaxTopDistInMatrixRow) {
        this.highestMaxTopDistInMatrixRow = highestMaxTopDistInMatrixRow;
    }

    public void setDescriptor(String key, String value) {
        this.descriptorResultMap.put(key, value);
    }

    public String getDescriptor(String key) {
        return this.descriptorResultMap.get(key);
    }

    public TreeMap<String, String> getDescriptorResultMap() {
        return descriptorResultMap;
    }

}
