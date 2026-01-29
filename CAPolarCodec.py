import numpy as np
from python_polar_coding.polar_codes import SCListPolarCodec
from crc import Calculator, Crc16, Crc8, Configuration

# --- Define Calculators ---
# CRC-8 (CCITT)
CRC8_POLY = Crc8.CCITT
crc8_calculator = Calculator(CRC8_POLY)

# CRC-16 (XMODEM)
CRC16_POLY = Crc16.XMODEM
crc16_calculator = Calculator(CRC16_POLY)

# CRC-24 (LTE-A Standard)
config_24a = Configuration(
    width=24,
    polynomial=0x864CFB,      # x^24 + x^23 + x^18 + x^17 + x^14 + x^11 + x^10 + x^7 + x^6 + x^5 + x^4 + x^3 + 1
    init_value=0x000000,
    final_xor_value=0x000000,
    reverse_input=False,
    reverse_output=False
)
crc24_calculator = Calculator(config_24a)


class CAPolarCodec:
    """
    A wrapper class to implement CRC-Aided Polar coding using the SCLPolarCodec
    """
    def __init__(self, N: int, K_info: int, crc_len: int, list_size: int):
        """
        Initializes the CA-SCL Polar Codec.

        Args:
            N: The mother code block length (power of 2).
            K_info: The number of information bits per block.
            crc_len: The length of the CRC in bits.
            design_snr: The SNR in dB for which the code is constructed:
                    ---------------------------------------------------------------
                    ** Actual SNR range **          ** Recommended design_SNR **
                            <0 dB                               0 dB
                            0~2 dB                              2 dB
                            2~4 dB                              3-4 dB
                            >5 dB                               5 dB
                    ---------------------------------------------------------------
            list_size: The list size (L) for the SCL decoder.
        """
        self.N = N
        self.K_info = K_info
        self.crc_len = crc_len
        self.K_total = K_info + crc_len
        self.list_size = list_size
        design_snr = np.array([0, 2, 4, 5])  # '设计SNR'是为了给出构造Polar code阶段的信道排序依据，以选择冻结比特

        # --- Select Calculator Dynamically ---
        if crc_len == 8:
            self.crc_calculator = crc8_calculator
        elif crc_len == 16:
            self.crc_calculator = crc16_calculator
        elif crc_len == 24:
            self.crc_calculator = crc24_calculator
        else:
            raise ValueError(f"CRC length {self.crc_len} is not supported. Use 8, 16, and 24 bits.")

        # initialize the Polar codec
        self.polar_codecs = SCListPolarCodec(
            N=self.N,
            K=self.K_total,
            design_snr=design_snr[1],
            L=self.list_size
        )

    def encode(self, info_bits: np.ndarray):
        """
        Appends CRC and then Polar encodes the message.

        Args:
            info_bits: A NumPy array of K_info information bits.

        Returns:
            A NumPy array of N encoded bits.
        """
        # Convert bits to bytes for CRC calculation
        info_bytes = np.packbits(info_bits).tobytes()

        # Calculate CRC
        crc_val = self.crc_calculator.checksum(info_bytes)

        # Convert CRC value to bit array
        # e.g., 25 --> b:'11001' --> array([0,0,0,1,1,0,0,1]) when crc_len=8
        crc_bits = np.array([int(b) for b in f'{crc_val:0{self.crc_len}b}'])

        # Append CRC to information bits
        message_with_crc = np.concatenate((info_bits, crc_bits))

        # Polar encode the combined message
        encoder_bits = self.polar_codecs.encode(message_with_crc)

        return encoder_bits

    def decode(self, received_llrs: np.ndarray):
        """
        Performs CA-SCL decoding.

        Args:
            received_llrs: A NumPy array of N log-likelihood ratios.

        Returns:
            A tuple containing the decoded information bits (K_info bits) and a boolean
            indicating if the CRC check passed.
        """
        # SCL decoding returns a list of L candidate messages
        candidate_messages = self.polar_codecs.decode(received_llrs)

        # Iterate through the list to find the first valid CRC

        decoded_info_bits = candidate_messages[:self.K_info]
        decoded_crc_bits = candidate_messages[self.K_info:]

        # Convert decoded info bits to bytes to calculate expected CRC
        decoded_info_bytes = np.packbits(decoded_info_bits).tobytes()
        expected_crc_val = self.crc_calculator.checksum(decoded_info_bytes)

        # Convert decoded CRC bits to integer value
        decoded_crc_val = int("".join(map(str, decoded_crc_bits)), 2)

        if expected_crc_val == decoded_crc_val:
            # CRC passed, this is our message
            return decoded_info_bits, True

        bestguess_info_bits = candidate_messages[:self.K_info]
        return bestguess_info_bits, False

