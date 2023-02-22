import random


def bootstrap_snp_level(map_in, map_out, ped_in, ped_out, seed):
    # sample the SNPs using the map file
    # each line in the map file corresponds to one SNP
    random.seed(seed)
    num_samples = sum(1 for _ in open(map_in))
    bootstrap_snp_indices = sorted(random.choices(range(num_samples), k=num_samples))

    # 1. Bootstrap the .map file
    snps = open(map_in).readlines()
    seen_snps = set()

    with open(map_out, "a") as f:
        for bootstrap_idx in bootstrap_snp_indices:
            snp_line = snps[bootstrap_idx]

            # lines look like this:
            # 1   S_Adygei-1.DG 0 0 1 1
            chrom_id, snp_id, rest = snp_line.split(maxsplit=2)
            deduplicate = snp_id

            snp_id_counter = 0

            while deduplicate in seen_snps:
                snp_id_counter += 1
                deduplicate = f"{snp_id}_r{snp_id_counter}"

            seen_snps.add(deduplicate)
            f.write(f"{chrom_id} {deduplicate} {rest}")

    # 2. Bootstrap the .ped file using the bootstrap_snp_indices above
    # the .ped file contains one line for each individual sample
    # each line has 2V + 6 fields with V being the number of samples
    # The first six fields do not change
    ped_out_handle = open(ped_out, "a")

    for indiv_line in open(ped_in):
        indiv_line = indiv_line.strip()
        fields = indiv_line.split()

        # the first 6 fields don't change with bootstrapping
        new_indiv_line = fields[:6]

        # the following lines correspond to the SNPs
        # each SNP accounts for two fields
        # so for each index in the bootstrap_snp_indices we have to access two fields:
        # 5 + (2 * (index + 1)) and 5 + (2 * (index + 1) + 1)
        # (5 and (index + 1) since Python is 0-indexed)
        for bootstrap_idx in bootstrap_snp_indices:
            idx_var1 = 5 + 2 * (bootstrap_idx + 1)
            new_indiv_line.append(fields[idx_var1 - 1])
            new_indiv_line.append(fields[idx_var1])

        ped_out_handle.write(" ".join(new_indiv_line))
        ped_out_handle.write("\n")

    ped_out_handle.close()


def bootstrap_indiv_level():
    raise NotImplementedError()

