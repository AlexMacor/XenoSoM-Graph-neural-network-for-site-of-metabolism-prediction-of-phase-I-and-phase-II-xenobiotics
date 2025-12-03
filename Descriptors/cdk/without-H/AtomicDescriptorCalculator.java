import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.atomtype.SybylAtomTypeMatcher;
import org.openscience.cdk.graph.ShortestPaths;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IAtomType;
import org.openscience.cdk.io.SDFWriter;
import org.openscience.cdk.io.iterator.IteratingSDFReader;
import org.openscience.cdk.qsar.AbstractAtomicDescriptor;
import org.openscience.cdk.qsar.DescriptorValue;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.ArrayList;

public class AtomicDescriptorCalculator {

    /* AtomicDescriptor 種別の定義 */
    public static HashMap<String, AbstractAtomicDescriptor> atomicDescriptorMap = new HashMap<String, AbstractAtomicDescriptor>();

    static {
        atomicDescriptorMap.put("EffectiveAtomPolarizability",
                new org.openscience.cdk.qsar.descriptors.atomic.EffectiveAtomPolarizabilityDescriptor());
        atomicDescriptorMap.put("StabilizationPlusCharge",
                new org.openscience.cdk.qsar.descriptors.atomic.StabilizationPlusChargeDescriptor());
        atomicDescriptorMap.put("SigmaElectronegativity",
                new org.openscience.cdk.qsar.descriptors.atomic.SigmaElectronegativityDescriptor());
        atomicDescriptorMap.put("PiElectronegativity",
                new org.openscience.cdk.qsar.descriptors.atomic.PiElectronegativityDescriptor());
        atomicDescriptorMap.put("PartialSigmaCharge",
                new org.openscience.cdk.qsar.descriptors.atomic.PartialSigmaChargeDescriptor());
        atomicDescriptorMap.put("PartialTChargeMMFF94",
                new org.openscience.cdk.qsar.descriptors.atomic.PartialTChargeMMFF94Descriptor());
        atomicDescriptorMap.put("AtomDegree", new org.openscience.cdk.qsar.descriptors.atomic.AtomDegreeDescriptor());
        atomicDescriptorMap.put("AtomValance", new org.openscience.cdk.qsar.descriptors.atomic.AtomValenceDescriptor());
        atomicDescriptorMap.put("AtomHybridizationVSEPR",
                new org.openscience.cdk.qsar.descriptors.atomic.AtomHybridizationVSEPRDescriptor());
        atomicDescriptorMap.put("AtomHybridization",
                new org.openscience.cdk.qsar.descriptors.atomic.AtomHybridizationDescriptor());
    }

    public static void main(String args[]) throws IOException{

        // 引数のチェック
        if (args.length != 3) {
            System.err.println("AtomicCircleDescriptorCalculator <input mol or sdf> <output mol or sdf> <output csv>");
            System.exit(1);
        }

        FileInputStream fis = null;
        IteratingSDFReader isr = null;
        FileWriter sdfFr = new FileWriter(args[1]);
        SDFWriter sdfWriter = new SDFWriter(sdfFr);
        ArrayList<AtomInfo> atomInfoList = new ArrayList<AtomInfo>();

        try {
            fis = new FileInputStream(new File(args[0]));
            isr = new IteratingSDFReader(fis, DefaultChemObjectBuilder.getInstance());

            // Syby AtomTypeMatcher の定義
            SybylAtomTypeMatcher typeMatcher = SybylAtomTypeMatcher.getInstance(DefaultChemObjectBuilder.getInstance());

            int cnt = 1;
            // 各分子の読み込み
            while (isr.hasNext()) {

                IAtomContainer mol = (IAtomContainer) isr.next();

                System.out.println("*** " + (cnt++) + " ***" + mol.getProperty("cdk:Title"));

                // 原子タイプの取得
                IAtomType[] atomTypes = typeMatcher.findMatchingAtomTypes(mol);

                // ShortestPaths格納用マトリクス
                int[][] shortestPaths = new int[mol.getAtomCount()][mol.getAtomCount()];

                // 最大パスの保持
                int longestMaxTopDistInMolecule = Integer.MIN_VALUE;

                // 分子毎のAtom情報
                AtomInfo[] atomInfoArrayByMolecule = new AtomInfo[mol.getAtomCount()];

                // 各原子の読み込みと原子記述子の計算、格納
                for (IAtom atom : mol.atoms()) {
                    // AtomInfoの生成
                    AtomInfo atomInfo = new AtomInfo();
                    atomInfoArrayByMolecule[atom.getIndex()] = atomInfo;
                    atomInfo.setTitle(mol.getProperty("cdk:Title"));
                    // atomInfo.setIndex(atom.getIndex()+1);
                    atomInfo.setIndex(atom.getIndex());
                    atomInfo.setSymbol(atom.getSymbol());
                    if (atomTypes[atom.getIndex()] != null) {
                        atomInfo.setAtomType(atomTypes[atom.getIndex()].getAtomTypeName());
                    }

                    // 原子記述子計算
                    for (String descriptorName : atomicDescriptorMap.keySet()) {
                        AbstractAtomicDescriptor aad = atomicDescriptorMap.get(descriptorName);
                        DescriptorValue dv = aad.calculate(atom, mol);
                        String aadValue = dv.getValue().toString();
                        atomInfo.setDescriptor(descriptorName, aadValue);
                    }

                    // パスに関する記述子の計算
                    ShortestPaths sp = new ShortestPaths(mol, atom);
                    atomInfo.setHighestMaxTopDistInMatrixRow(Integer.MIN_VALUE);
                    for (IAtom atom_target : mol.atoms()) {
                        shortestPaths[atom.getIndex()][atom_target.getIndex()] = sp.distanceTo(atom_target);
                        // 水素は無視する
                        if (!atom.getSymbol().equals("H") && !atom_target.getSymbol().equals("H")) {
                            if (sp.distanceTo(atom_target) > atomInfo.getHighestMaxTopDistInMatrixRow()) {
                                atomInfo.setHighestMaxTopDistInMatrixRow(sp.distanceTo(atom_target));
                            }
                            if (sp.distanceTo(atom_target) > longestMaxTopDistInMolecule) {
                                longestMaxTopDistInMolecule = sp.distanceTo(atom_target);
                            }
                        }
                    }
                }

                // 分子中の最大パスの設定
                for (int i = 0; i < shortestPaths.length; i++) {
                    atomInfoArrayByMolecule[i].setLongestMaxTopDistInMolecule(longestMaxTopDistInMolecule);
                }

                // 各原子の登録
                for (int i = 0; i < atomInfoArrayByMolecule.length; i++) {
                    if (!atomInfoArrayByMolecule[i].getSymbol().equals("H")) {
                        atomInfoList.add(atomInfoArrayByMolecule[i]);
                    }
                }

                // 各原子の記述子データを集めてmolオブジェクトに登録
                String longestMaxTopInMoleculeStr = "";
                String highestMaxTopInMoleculeStr = "";
                String diffSPAN3Str = "";
                String relSPAN4Str = "";
                for (int i = 0; i < atomInfoArrayByMolecule.length; i++) {
                    if(i != 0){
                        longestMaxTopInMoleculeStr += " ";
                        highestMaxTopInMoleculeStr += " ";
                        diffSPAN3Str += " ";
                        relSPAN4Str += " ";
                    }
                    longestMaxTopInMoleculeStr += atomInfoArrayByMolecule[i].getLongestMaxTopDistInMolecule();
                    highestMaxTopInMoleculeStr += atomInfoArrayByMolecule[i].getHighestMaxTopDistInMatrixRow();
                    diffSPAN3Str += atomInfoArrayByMolecule[i].getLongestMaxTopDistInMolecule() - atomInfoArrayByMolecule[i].getHighestMaxTopDistInMatrixRow();
                    relSPAN4Str += String.valueOf((double) atomInfoArrayByMolecule[i].getHighestMaxTopDistInMatrixRow() / (double) atomInfoArrayByMolecule[i].getLongestMaxTopDistInMolecule());
                }
                mol.setProperty("longestMaxTopInMolecule", longestMaxTopInMoleculeStr);
                mol.setProperty("highestMaxTopInMolecule", highestMaxTopInMoleculeStr);
                mol.setProperty("diffSPAN3", diffSPAN3Str);
                mol.setProperty("relSPAN4", relSPAN4Str);

                for (String descriptorName : atomicDescriptorMap.keySet()) {
                    String data = "";
                    for (int i = 0; i < atomInfoArrayByMolecule.length; i++) {
                        if(i != 0){
                            data += " ";
                        }
                        data += (String)atomInfoArrayByMolecule[i].getDescriptor(descriptorName);
                    }
                    mol.setProperty(descriptorName, data);
                }
                sdfWriter.write(mol);
            }

            sdfWriter.close();
            sdfFr.close();

            // 全分子/原子についてCSV出力
            FileWriter fw = new FileWriter(args[2]);

            // ヘッダ出力
            fw.write("Title,");
            fw.write("Index,");
            fw.write("Symbol,");
            fw.write("AtomType,");

            fw.write("longestMaxTopDistInMolecule,");
            fw.write("highestMaxTopDistInMatrixRow,");
            fw.write("diffSPAN3,");
            fw.write("relSPAN4");

            // 記述子計算結果の出力
            for (String descriptorName : atomicDescriptorMap.keySet()) {
                fw.write("," );
                fw.write(descriptorName);
            }
            fw.write("\n");

            // データ出力
            for (AtomInfo atomInfo : atomInfoList) {
                // 名前
                fw.write(atomInfo.getTitle() + ",");
                // ID出力
                fw.write(atomInfo.getIndex() + ",");
                // 原子
                fw.write(atomInfo.getSymbol() + ",");
                // 種別
                fw.write(atomInfo.getAtomType() + ",");

                fw.write(atomInfo.getLongestMaxTopDistInMolecule() + ",");
                fw.write(atomInfo.getHighestMaxTopDistInMatrixRow() + ",");
                fw.write(
                        (atomInfo.getLongestMaxTopDistInMolecule() - atomInfo.getHighestMaxTopDistInMatrixRow()) + ",");
                fw.write( String.valueOf((double) atomInfo.getHighestMaxTopDistInMatrixRow() / (double) atomInfo.getLongestMaxTopDistInMolecule()));

                // 記述子計算結果の出力
                for (String descriptorName : atomicDescriptorMap.keySet()) {
                    fw.write(",");
                    fw.write(atomInfo.getDescriptor(descriptorName));
                }
                fw.write("\n");
            }
            fw.close();

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            // ファイルのクローズ処理
            if (isr != null) {
                try {
                    isr.close();
                } catch (IOException e2) {
                    e2.printStackTrace();
                    ;
                }
            }
            if (fis != null) {
                try {
                    fis.close();
                } catch (IOException e2) {
                    e2.printStackTrace();
                }
            }
        }
    }
}
